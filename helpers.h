// Macro for concatenation
#define CONCATENATE(arg1, arg2)   CONCATENATE1(arg1, arg2)
#define CONCATENATE1(arg1, arg2)  arg1 ## arg2

// Print Macro
#define PRINT_VAR(var) print_str(#var " = ");printIfPrintable(var);

// Helper macros to count the number of arguments
#define ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define N_ARGS(...) ARG_N(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

// Macros to expand each argument for PRINT
#define PRINT_VAR_1(a) PRINT_VAR(a)
#define PRINT_VAR_2(a, b) PRINT_VAR(a) PRINT_VAR(b)
#define PRINT_VAR_3(a, b, c) PRINT_VAR(a) PRINT_VAR_2(b, c)
#define PRINT_VAR_4(a, b, c, d) PRINT_VAR(a) PRINT_VAR_3(b, c, d)
#define PRINT_VAR_5(a, b, c, d, e) PRINT_VAR(a) PRINT_VAR_4(b, c, d, e)
#define PRINT_VAR_6(a, b, c, d, e, f) PRINT_VAR(a) PRINT_VAR_5(b, c, d, e, f)
#define PRINT_VAR_7(a, b, c, d, e, f, g) PRINT_VAR(a) PRINT_VAR_6(b, c, d, e, f, g)
#define PRINT_VAR_8(a, b, c, d, e, f, g, h) PRINT_VAR(a) PRINT_VAR_7 (b, c, d, e, f, g, h)
#define PRINT_VAR_9(a, b, c, d, e, f, g, h, i) PRINT_VAR(a) PRINT_VAR_8(b, c, d, e, f, g, h, i)
#define PRINT_VAR_10(a, b, c, d, e, f, g, h, i, j) PRINT_VAR(a) PRINT_VAR_9(b, c, d, e, f, g, h, i, j)

// Macro to select the correct PRINT_VAR_X macro based on the number of arguments
#define EXPAND_ARGS_HELPER(N, ...) CONCATENATE(PRINT_VAR_, N)(__VA_ARGS__)
#define EXPAND_ARGS(...) EXPAND_ARGS_HELPER(N_ARGS(__VA_ARGS__), __VA_ARGS__)
