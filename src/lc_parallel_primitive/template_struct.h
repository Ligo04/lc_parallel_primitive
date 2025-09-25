#pragma once
#include <luisa/dsl/struct.h>
#include <luisa/core/macro.h>

#define LUISA_STRUCTURE_MAP_MEMBER_TO_DESC_TEMPLATE(m, S)                      \
    luisa::compute::detail::TypeDesc<                                          \
        std::remove_cvref_t<decltype(std::declval<S()>().m)>>::description()
#define LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE_TEMPLATE(m, S)                      \
    typename std::remove_cvref_t<decltype(std::declval<S()>().m)>

#ifdef _MSC_VER  // force the built-in offsetof(), otherwise clangd would complain that it's not constant
#define LUISA_STRUCTURE_MAP_MEMBER_TO_OFFSET_TEMPLATE(m, S)                    \
    __builtin_offsetof(S(), m)
#else
#define LUISA_STRUCTURE_MAP_MEMBER_TO_OFFSET_TEMPLATE(m, S) offsetof(S(), m)
#endif


#define LUISA_STRUCT_MAKE_MEMBER_TYPE_TEMPLATE(m, S)                           \
    using member_type_##m = detail::c_array_to_std_array_t<                    \
        std::remove_cvref_t<decltype(std::declval<S()>().m)>>;


#define LUISA_MAP0_UD(f, userdata, x, peek, ...)                               \
    f(x, userdata)                                                             \
        LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP1_UD))(f, userdata, peek, __VA_ARGS__)
#define LUISA_MAP1_UD(f, userdata, x, peek, ...)                               \
    f(x, userdata)                                                             \
        LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP0_UD))(f, userdata, peek, __VA_ARGS__)

#define LUISA_MAP_LIST0_UD(f, userdata, x, peek, ...)                          \
    , f(x, userdata) LUISA_MACRO_DEFER(                                        \
          LUISA_MAP_NEXT(peek, LUISA_MAP_LIST1_UD))(f, userdata, peek, __VA_ARGS__)
#define LUISA_MAP_LIST1_UD(f, userdata, x, peek, ...)                          \
    , f(x, userdata) LUISA_MACRO_DEFER(                                        \
          LUISA_MAP_NEXT(peek, LUISA_MAP_LIST0_UD))(f, userdata, peek, __VA_ARGS__)
#define LUISA_MAP_LIST2_UD(f, userdata, x, peek, ...)                          \
    f(x, userdata) LUISA_MACRO_DEFER(                                          \
        LUISA_MAP_NEXT(peek, LUISA_MAP_LIST1_UD))(f, userdata, peek, __VA_ARGS__)

#define LUISA_MAP_LIST0_UD_I(f, userdata, index, x, peek, ...)                 \
    , f(x, userdata, index)                                                    \
          LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP_LIST1_UD_I))(       \
              f, userdata, LUISA_INC(index), peek, __VA_ARGS__)
#define LUISA_MAP_LIST1_UD_I(f, userdata, index, x, peek, ...)                 \
    , f(x, userdata, index)                                                    \
          LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP_LIST0_UD_I))(       \
              f, userdata, LUISA_INC(index), peek, __VA_ARGS__)
#define LUISA_MAP_LIST2_UD_I(f, userdata, index, x, peek, ...)                           \
    f(x, userdata, index) LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP_LIST1_UD_I))( \
        f, userdata, LUISA_INC(index), peek, __VA_ARGS__)

// Applies the function macro `f` to each of the remaining parameters, inserts commas between the results,
// and passes userdata as the second parameter to each invocation,
// e.g. LUISA_MAP_LIST_UD(f, x, a, b, c) evaluates to f(a, x), f(b, x), f(c, x)

#define LUISA_MAP_LIST_UD(f, userdata, ...)                                    \
    LUISA_MACRO_EVAL(                                                          \
        LUISA_MAP_LIST2_UD(f, userdata, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

// Applies the function macro `f` to each of the remaining parameters and passes userdata as the second parameter to each invocation,
// LUISA_MAP_UD(f, x, a, b, c) evaluates to f(a, x) f(b, x) f(c, x)

#define LUISA_MAP_UD(f, userdata, ...)                                         \
    LUISA_MACRO_EVAL(LUISA_MAP1_UD(f, userdata, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

#define LUISA_DERIVE_FMT_TEMPLATE(TEMPLATE, STRUCT, DisplayName, ...)                                    \
    TEMPLATE()                                                                                           \
    struct fmt::formatter<STRUCT()>                                                                      \
    {                                                                                                    \
        constexpr auto parse(format_parse_context& ctx) const -> decltype(ctx.begin())                   \
        {                                                                                                \
            return ctx.end();                                                                            \
        }                                                                                                \
        template <typename FormatContext>                                                                \
        auto format(const STRUCT() & input, FormatContext& ctx) const                                    \
            -> decltype(ctx.out())                                                                       \
        {                                                                                                \
            return fmt::format_to(ctx.out(),                                                             \
                                  FMT_STRING(#DisplayName "{{ {} }}"),                                   \
                                  fmt::join(std::array{LUISA_MAP_LIST(LUISA_DERIVE_FMT_MAP_STRUCT_FIELD, \
                                                                      __VA_ARGS__)},                     \
                                            ", "));                                                      \
        }                                                                                                \
    };


#define LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION_TEMPLATE(TEMPLATE, S, ...)                                \
    TEMPLATE()                                                                                                  \
    struct luisa::compute::struct_member_tuple<S()>                                                             \
    {                                                                                                           \
        using this_type = S();                                                                                  \
        using type =                                                                                            \
            std::tuple<LUISA_MAP_LIST(LUISA_STRUCTURE_MAP_MEMBER_TO_TYPE, ##__VA_ARGS__)>;                      \
        using offset =                                                                                          \
            std::integer_sequence<size_t, LUISA_MAP_LIST(LUISA_STRUCTURE_MAP_MEMBER_TO_OFFSET, ##__VA_ARGS__)>; \
        static_assert(alignof(S()) >= 4);                                                                       \
        static_assert(luisa::compute::detail::is_valid_reflection_v<S(), type, offset>);                        \
    };                                                                                                          \
    TEMPLATE()                                                                                                  \
    struct luisa::compute::detail::TypeDesc<S()>                                                                \
    {                                                                                                           \
        using this_type = S();                                                                                  \
        static luisa::string_view description() noexcept                                                        \
        {                                                                                                       \
            static auto s = luisa::compute::detail::make_struct_description(                                    \
                alignof(S()),                                                                                   \
                {LUISA_MAP_LIST_UD(LUISA_STRUCTURE_MAP_MEMBER_TO_DESC_TEMPLATE, S, ##__VA_ARGS__)});            \
            return s;                                                                                           \
        }                                                                                                       \
    };

#define LUISA_STRUCT_REFLECT_TEMPLATE(TEMPLATE, S, ...)                        \
    LUISA_MAKE_STRUCTURE_TYPE_DESC_SPECIALIZATION_TEMPLATE(TEMPLATE, S, __VA_ARGS__)

#define LUISA_DERIVE_DSL_STRUCT_TEMPLATE(TEMPLATE, S, ...)                                      \
    namespace luisa::compute                                                                    \
    {                                                                                           \
    namespace detail                                                                            \
    {                                                                                           \
        TEMPLATE()                                                                              \
        class AtomicRef<S()> : private AtomicRefBase                                            \
        {                                                                                       \
          private:                                                                              \
            using this_type = S();                                                              \
            LUISA_MAP_UD(LUISA_STRUCT_MAKE_MEMBER_TYPE_TEMPLATE, S, __VA_ARGS__)                \
            [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept \
            {                                                                                   \
                constexpr const std::string_view member_names[]{                                \
                    LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                              \
                return std::find(std::begin(member_names), std::end(member_names), name)        \
                       - std::begin(member_names);                                              \
            }                                                                                   \
                                                                                                \
          public:                                                                               \
            LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_ATOMIC_REF_DECL, __VA_ARGS__)                    \
            explicit AtomicRef(const AtomicRefNode* node) noexcept                              \
                : AtomicRefBase{node}                                                           \
            {                                                                                   \
            }                                                                                   \
        };                                                                                      \
    }                                                                                           \
    TEMPLATE()                                                                                  \
    struct Expr<S()> : public detail::ExprEnableBitwiseCast<Expr<S()>>                          \
    {                                                                                           \
      private:                                                                                  \
        using this_type = S();                                                                  \
        const Expression* _expression;                                                          \
        LUISA_MAP_UD(LUISA_STRUCT_MAKE_MEMBER_TYPE_TEMPLATE, S, __VA_ARGS__)                    \
        [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept     \
        {                                                                                       \
            constexpr const std::string_view member_names[]{                                    \
                LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                                  \
            return std::find(std::begin(member_names), std::end(member_names), name)            \
                   - std::begin(member_names);                                                  \
        }                                                                                       \
                                                                                                \
      public:                                                                                   \
        LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_EXPR_DECL, __VA_ARGS__)                              \
        explicit Expr(const Expression* e) noexcept                                             \
            : _expression{e}                                                                    \
            , LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__)                        \
        {                                                                                       \
        }                                                                                       \
        [[nodiscard]] auto expression() const noexcept                                          \
        {                                                                                       \
            return this->_expression;                                                           \
        }                                                                                       \
        Expr(Expr&& another) noexcept      = default;                                           \
        Expr(const Expr& another) noexcept = default;                                           \
        Expr& operator=(Expr) noexcept     = delete;                                            \
        template <size_t i>                                                                     \
        [[nodiscard]] auto get() const noexcept                                                 \
        {                                                                                       \
            using M = std::tuple_element_t<i, struct_member_tuple_t<S()>>;                      \
            return Expr<M>{detail::FunctionBuilder::current()->member(                          \
                Type::of<M>(), this->expression(), i)};                                         \
        };                                                                                      \
    };                                                                                          \
    namespace detail                                                                            \
    {                                                                                           \
        TEMPLATE()                                                                              \
        struct Ref<S()> : public detail::ExprEnableBitwiseCast<Ref<S()>>,                       \
                          public detail::RefEnableGetAddress<Ref<S()>>                          \
        {                                                                                       \
          private:                                                                              \
            using this_type = S();                                                              \
            const Expression* _expression;                                                      \
            LUISA_MAP_UD(LUISA_STRUCT_MAKE_MEMBER_TYPE_TEMPLATE, S, __VA_ARGS__)                \
            [[nodiscard]] static constexpr size_t _member_index(std::string_view name) noexcept \
            {                                                                                   \
                constexpr const std::string_view member_names[]{                                \
                    LUISA_MAP_LIST(LUISA_STRINGIFY, __VA_ARGS__)};                              \
                return std::find(std::begin(member_names), std::end(member_names), name)        \
                       - std::begin(member_names);                                              \
            }                                                                                   \
                                                                                                \
          public:                                                                               \
            LUISA_MAP(LUISA_STRUCT_MAKE_MEMBER_REF_DECL, __VA_ARGS__)                           \
            explicit Ref(const Expression* e) noexcept                                          \
                : _expression{e}                                                                \
                , LUISA_MAP_LIST(LUISA_STRUCT_MAKE_MEMBER_INIT, __VA_ARGS__)                    \
            {                                                                                   \
            }                                                                                   \
            [[nodiscard]] auto expression() const noexcept                                      \
            {                                                                                   \
                return this->_expression;                                                       \
            }                                                                                   \
            Ref(Ref&& another) noexcept      = default;                                         \
            Ref(const Ref& another) noexcept = default;                                         \
            [[nodiscard]] operator Expr<S()>() const noexcept                                   \
            {                                                                                   \
                return Expr<S()>{this->expression()};                                           \
            }                                                                                   \
            template <typename Rhs>                                                             \
            void operator=(Rhs&& rhs) & noexcept                                                \
            {                                                                                   \
                dsl::assign(*this, std::forward<Rhs>(rhs));                                     \
            }                                                                                   \
            void operator=(Ref rhs) & noexcept                                                  \
            {                                                                                   \
                (*this) = Expr{rhs};                                                            \
            }                                                                                   \
            template <size_t i>                                                                 \
            [[nodiscard]] auto get() const noexcept                                             \
            {                                                                                   \
                using M = std::tuple_element_t<i, struct_member_tuple_t<S()>>;                  \
                return Ref<M>{detail::FunctionBuilder::current()->member(                       \
                    Type::of<M>(), this->expression(), i)};                                     \
            };                                                                                  \
            [[nodiscard]] auto operator->() noexcept                                            \
            {                                                                                   \
                return reinterpret_cast<luisa_compute_extension<S()>*>(this);                   \
            }                                                                                   \
            [[nodiscard]] auto operator->() const noexcept                                      \
            {                                                                                   \
                return reinterpret_cast<const luisa_compute_extension<S()>*>(this);             \
            }                                                                                   \
        };                                                                                      \
    }                                                                                           \
    }

#define LUISA_DERIVE_SOA_VIEW_TEMPLATE(TEMPLATE, S, ...)                                          \
    namespace luisa::compute                                                                      \
    {                                                                                             \
    TEMPLATE()                                                                                    \
    class SOAView<S()> : public detail::SOAViewBase<S()>                                          \
    {                                                                                             \
                                                                                                  \
      private:                                                                                    \
        using this_type = S();                                                                    \
        LUISA_MAP_UD(LUISA_STRUCT_MAKE_MEMBER_TYPE_TEMPLATE, S, __VA_ARGS__)                      \
                                                                                                  \
      public:                                                                                     \
        [[nodiscard]] static auto compute_soa_size(auto soa_size) noexcept                        \
        {                                                                                         \
            return LUISA_MAP(LUISA_SOA_VIEW_MAKE_MEMBER_SOA_SIZE_ACCUM, __VA_ARGS__) 0u;          \
        }                                                                                         \
                                                                                                  \
      public:                                                                                     \
        LUISA_MAP(LUISA_SOA_VIEW_MAKE_MEMBER_DECL, __VA_ARGS__)                                   \
                                                                                                  \
      private:                                                                                    \
        template <typename T>                                                                     \
        [[nodiscard]] static auto _accumulate_soa_offset(size_t& accum, size_t soa_size) noexcept \
        {                                                                                         \
            auto offset = accum;                                                                  \
            accum += SOAView<T>::compute_soa_size(soa_size);                                      \
            return offset;                                                                        \
        }                                                                                         \
                                                                                                  \
        SOAView(size_t           soa_offset_accum,                                                \
                BufferView<uint> buffer,                                                          \
                size_t           soa_offset,                                                      \
                size_t           soa_size,                                                        \
                size_t           elem_offset,                                                     \
                size_t           elem_size) noexcept                                                        \
            : detail::SOAViewBase<S()>{buffer, soa_offset, soa_size, elem_offset, elem_size}      \
            , LUISA_MAP_LIST(LUISA_SOA_VIEW_MAKE_MEMBER_INIT, __VA_ARGS__)                        \
        {                                                                                         \
        }                                                                                         \
                                                                                                  \
      public:                                                                                     \
        SOAView(BufferView<uint> buffer,                                                          \
                size_t           soa_offset,                                                      \
                size_t           soa_size,                                                        \
                size_t           elem_offset,                                                     \
                size_t           elem_size) noexcept                                                        \
            : SOAView{0u, buffer, soa_offset, soa_size, elem_offset, elem_size}                   \
        {                                                                                         \
        }                                                                                         \
                                                                                                  \
      public:                                                                                     \
        using detail::SOAViewBase<S()>::operator->;                                               \
    };                                                                                            \
    }

#define LUISA_DERIVE_SOA_EXPR_TEMPLATE(TEMPLATE, S, ...)                                        \
    namespace luisa::compute                                                                    \
    {                                                                                           \
    TEMPLATE()                                                                                  \
    struct Expr<SOA<S()>> : public detail::SOAExprBase                                          \
    {                                                                                           \
      private:                                                                                  \
        using this_type = S();                                                                  \
        LUISA_MAP_UD(LUISA_STRUCT_MAKE_MEMBER_TYPE_TEMPLATE, S, __VA_ARGS__)                    \
                                                                                                \
      public:                                                                                   \
        LUISA_MAP(LUISA_SOA_EXPR_MAKE_MEMBER_DECL, __VA_ARGS__)                                 \
                                                                                                \
      private:                                                                                  \
        template <typename T>                                                                   \
        [[nodiscard]] static auto _accumulate_soa_offset(Var<uint>& accum,                      \
                                                         Expr<uint> soa_size) noexcept          \
        {                                                                                       \
            auto offset = accum;                                                                \
            accum += SOA<T>::compute_soa_size(soa_size);                                        \
            return offset;                                                                      \
        }                                                                                       \
        Expr(Var<uint>          soa_offset_accum,                                               \
             Expr<Buffer<uint>> buffer,                                                         \
             Expr<uint>         soa_offset,                                                     \
             Expr<uint>         soa_size,                                                       \
             Expr<uint>         elem_offset) noexcept                                                   \
            : detail::SOAExprBase{buffer, soa_offset, soa_size, elem_offset}                    \
            , LUISA_MAP_LIST(LUISA_SOA_EXPR_MAKE_MEMBER_INIT, __VA_ARGS__)                      \
        {                                                                                       \
        }                                                                                       \
                                                                                                \
      public:                                                                                   \
        Expr(Expr<Buffer<uint>> buffer,                                                         \
             Expr<uint>         soa_offset,                                                     \
             Expr<uint>         soa_size,                                                       \
             Expr<uint>         elem_offset) noexcept                                                   \
            : Expr{def(0u), buffer, soa_offset, soa_size, elem_offset}                          \
        {                                                                                       \
        }                                                                                       \
                                                                                                \
        Expr(SOAView<S()> soa) noexcept                                                         \
            : Expr{soa.buffer(), soa.soa_offset(), soa.soa_size(), soa.element_offset()}        \
        {                                                                                       \
        }                                                                                       \
                                                                                                \
        Expr(const SOA<S()>& soa) noexcept                                                      \
            : Expr{soa.view()}                                                                  \
        {                                                                                       \
        }                                                                                       \
                                                                                                \
        template <typename I>                                                                   \
        [[nodiscard]] auto read(I&& index) const noexcept                                       \
        {                                                                                       \
            auto i = dsl::def(std::forward<I>(index));                                          \
            return dsl::def<S()>(LUISA_MAP_LIST(LUISA_SOA_EXPR_MAKE_MEMBER_READ, __VA_ARGS__)); \
        }                                                                                       \
                                                                                                \
        template <typename I>                                                                   \
        [[nodiscard]] auto write(I&& index, Expr<S()> value) const noexcept                     \
        {                                                                                       \
            auto i = dsl::def(std::forward<I>(index));                                          \
            LUISA_MAP(LUISA_SOA_EXPR_MAKE_MEMBER_WRITE, __VA_ARGS__)                            \
        }                                                                                       \
                                                                                                \
        [[nodiscard]] auto operator->() const noexcept                                          \
        {                                                                                       \
            return this;                                                                        \
        }                                                                                       \
    };                                                                                          \
    }

#define LUISA_DERIVE_SOA_TEMPLATE(TEMPLATE, S, ...)                            \
    LUISA_DERIVE_SOA_VIEW_TEMPLATE(TEMPLATE, S, __VA_ARGS__)                   \
    LUISA_DERIVE_SOA_EXPR_TEMPLATE(TEMPLATE, S, __VA_ARGS__)


#define LUISA_TEMPLATE_STRUCT(TEMPLATE, S, ...)                                \
    LUISA_DERIVE_FMT_TEMPLATE(TEMPLATE, S, S, __VA_ARGS__)                     \
    LUISA_STRUCT_REFLECT_TEMPLATE(TEMPLATE, S, __VA_ARGS__)                    \
    TEMPLATE()                                                                 \
    struct luisa_compute_extension<S()>;                                       \
    LUISA_DERIVE_DSL_STRUCT_TEMPLATE(TEMPLATE, S, __VA_ARGS__)                 \
    LUISA_DERIVE_SOA_TEMPLATE(TEMPLATE, S, __VA_ARGS__)                        \
    TEMPLATE()                                                                 \
    struct luisa_compute_extension<S()> final : luisa::compute::detail::Ref<S()>