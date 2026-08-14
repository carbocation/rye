#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#endif

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

extern SEXP rye_simd_level(void);
extern SEXP rye_optimizer_abi(void);
extern SEXP rye_nnls_batch(SEXP, SEXP, SEXP, SEXP);
extern SEXP rye_gibbs_native_v2(
    SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP
);

static const R_CallMethodDef CallEntries[] = {
    {"rye_simd_level", (DL_FUNC) &rye_simd_level, 0},
    {"rye_optimizer_abi", (DL_FUNC) &rye_optimizer_abi, 0},
    {"rye_nnls_batch", (DL_FUNC) &rye_nnls_batch, 4},
    {"rye_gibbs_native_v2", (DL_FUNC) &rye_gibbs_native_v2, 9},
    {NULL, NULL, 0}
};

void attribute_visible R_init_rye(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
