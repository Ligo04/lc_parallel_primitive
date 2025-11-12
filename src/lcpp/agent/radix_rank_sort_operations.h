/*
 * @Author: Ligo 
 * @Date: 2025-11-07 10:52:00 
 * @Last Modified by: Ligo
 * @Last Modified time: 2025-11-12 23:48:51
 */

#pragma once

#include <luisa/dsl/var.h>
#include <luisa/dsl/struct.h>
#include <lcpp/common/type_trait.h>
#include <lcpp/common/util_type.h>

namespace luisa::parallel_primitive
{
using namespace luisa::compute;
template <typename KeyT, bool IsFP = std::is_floating_point_v<KeyT>>
struct BaseDigitExtractor
{
    using TraitsT      = Traits<KeyT>;
    using UnsignedBits = typename TraitsT::UnsignedBits;

    static compute::Var<UnsignedBits> ProcessFloatMinusZero(const compute::Var<UnsignedBits>& key)
    {
        return key;
    }
};

template <typename KeyT>
struct BaseDigitExtractor<KeyT, true>
{
    using TraitsT      = Traits<KeyT>;
    using UnsignedBits = typename TraitsT::UnsignedBits;

    static compute::Var<UnsignedBits> ProcessFloatMinusZero(const compute::Var<UnsignedBits>& key)
    {
        compute::Var<UnsignedBits> TWIDDLED_MINUS_ZERO_BITS = TraitsT::TwiddleIn(
            compute::Var<UnsignedBits>(1) << compute::Var<UnsignedBits>(8 * sizeof(UnsignedBits) - 1));
        compute::Var<UnsignedBits> TWIDDLED_ZERO_BITS = TraitsT::TwiddleIn(0);
        return key == TWIDDLED_MINUS_ZERO_BITS ? TWIDDLED_ZERO_BITS : key;
    }
};

template <typename KeyT>
struct ShiftDigitExtractor : BaseDigitExtractor<KeyT>
{
    using typename BaseDigitExtractor<KeyT>::UnsignedBits;

    compute::UInt bit_start;
    compute::UInt mask;

    explicit ShiftDigitExtractor(compute::UInt bit_start = 0, compute::UInt num_bits = 0)
        : bit_start(bit_start)
        , mask((1 << num_bits) - 1)
    {
    }

    compute::UInt Digit(const compute::Var<UnsignedBits>& key) const
    {
        return compute::UInt(this->ProcessFloatMinusZero(key) >> compute::Var<UnsignedBits>(bit_start)) & mask;
    }
};

namespace details
{
    namespace radix
    {
        template <class T, class = void>
        struct is_fundamental_type
        {
            static constexpr bool value = false;
        };

        template <class T>
        struct bit_ordered_conversion_policy_t
        {
            using bit_ordered_type = typename Traits<T>::UnsignedBits;

            static Var<bit_ordered_type> to_bit_ordered(const Var<bit_ordered_type>& val)
            {
                return Traits<T>::TwiddleIn(val);
            }

            static Var<bit_ordered_type> from_bit_ordered(const Var<bit_ordered_type>& val)
            {
                return Traits<T>::TwiddleOut(val);
            }
        };

        template <class T>
        struct bit_ordered_inversion_policy_t
        {
            using bit_ordered_type = typename Traits<T>::UnsignedBits;

            static Var<bit_ordered_type> inverse(const Var<bit_ordered_type>& val)
            {
                return Var<bit_ordered_type>(~val);
            }
        };

        template <class T, bool = is_fundamental_type<T>::value>
        struct traits_t
        {
            using bit_ordered_type              = typename Traits<T>::UnsignedBits;
            using bit_ordered_conversion_policy = bit_ordered_conversion_policy_t<T>;
            using bit_ordered_inversion_policy  = bit_ordered_inversion_policy_t<T>;

            template <class FundamentalExtractorT>
            using digit_extractor_t = FundamentalExtractorT;

            static Var<bit_ordered_type> min_raw_binary_key()
            {
                return Var<bit_ordered_type>(Traits<T>::LOWEST_KEY);
            }

            static Var<bit_ordered_type> max_raw_binary_key()
            {
                return Var<bit_ordered_type>(Traits<T>::MAX_KEY);
            }

            static Int default_end_bit() { return Int(sizeof(T) * 8); }

            template <class FundamentalExtractorT>
            static digit_extractor_t<FundamentalExtractorT> digit_extractor(int begin_bit, int num_bits)
            {
                return FundamentalExtractorT(begin_bit, num_bits);
            }
        };
    }  // namespace radix
}  // namespace details


template <bool IS_DESCENDING, NumericT KeyType>
struct RadixSortTwiddle
{
  private:
    using traits                        = details::radix::traits_t<KeyType>;
    using bit_ordered_type              = typename traits::bit_ordered_type;
    using bit_ordered_conversion_policy = typename traits::bit_ordered_conversion_policy;
    using bit_ordered_inversion_policy  = typename traits::bit_ordered_inversion_policy;

  public:
    static Var<bit_ordered_type> In(Var<bit_ordered_type> key)
    {
        key = bit_ordered_conversion_policy::to_bit_ordered(key);
        if constexpr(IS_DESCENDING)
        {
            key = bit_ordered_inversion_policy::inverse(key);
        }
        return key;
    }

    static Var<bit_ordered_type> Out(Var<bit_ordered_type> key)
    {
        if constexpr(IS_DESCENDING)
        {
            key = bit_ordered_inversion_policy::inverse(key);
        }
        key = bit_ordered_conversion_policy::from_bit_ordered(key);
        return key;
    }
    static Var<bit_ordered_type> DefaultKey()
    {
        return IS_DESCENDING ? traits::min_raw_binary_key() : traits::max_raw_binary_key();
    }
};
}  // namespace luisa::parallel_primitive