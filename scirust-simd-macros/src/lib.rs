use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, ItemFn, parse_macro_input};

/// Attribute macro that generates architecture-specific SIMD variants of a free
/// function and a runtime dispatcher.
///
/// For a function `foo`, the macro emits:
///
/// * `__simd_scirust_foo_avx2`  – x86/x86_64, `#[target_feature(enable = "avx2")]`
/// * `__simd_scirust_foo_sse2`  – x86/x86_64, `#[target_feature(enable = "sse2")]`
/// * `__simd_scirust_foo_neon`  – aarch64,   `#[target_feature(enable = "neon")]`
/// * `__simd_scirust_foo_scalar` – scalar fallback
/// * `foo`                       – public dispatcher
///
/// # Limitations
/// * Works best on free functions with simple identifier arguments.
/// * Does not support `const fn` or `async fn` because `#[target_feature]` is
///   incompatible with those.
#[proc_macro_attribute]
pub fn simd(_args: TokenStream, input: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(input as ItemFn);

    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let block = &input_fn.block;
    let fn_name = &sig.ident;

    // Collect argument names so we can forward them in the dispatcher.
    let arg_names: Vec<_> = sig
        .inputs
        .iter()
        .map(|arg| match arg
        {
            FnArg::Receiver(_) => quote!(self),
            FnArg::Typed(pat_type) =>
            {
                let pat = &pat_type.pat;
                quote!(#pat)
            },
        })
        .collect();

    // Decompose the original signature for reconstruction.
    let constness = &sig.constness;
    let asyncness = &sig.asyncness;
    let unsafety = &sig.unsafety;
    let abi = &sig.abi;
    let generics = &sig.generics;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let where_clause = &sig.generics.where_clause;

    let avx2_name = quote::format_ident!("__simd_scirust_{}_avx2", fn_name);
    let sse2_name = quote::format_ident!("__simd_scirust_{}_sse2", fn_name);
    let neon_name = quote::format_ident!("__simd_scirust_{}_neon", fn_name);
    let scalar_name = quote::format_ident!("__simd_scirust_{}_scalar", fn_name);

    let expanded = quote! {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #constness #asyncness unsafe #abi fn #avx2_name #generics(#inputs) #output #where_clause #block

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "sse2")]
        #constness #asyncness unsafe #abi fn #sse2_name #generics(#inputs) #output #where_clause #block

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #constness #asyncness unsafe #abi fn #neon_name #generics(#inputs) #output #where_clause #block

        #constness #asyncness #unsafety #abi fn #scalar_name #generics(#inputs) #output #where_clause #block

        #vis #constness #asyncness #unsafety #abi fn #fn_name #generics(#inputs) #output #where_clause {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if ::std::arch::is_x86_feature_detected!("avx2") {
                    unsafe { #avx2_name(#(#arg_names),*) }
                } else if ::std::arch::is_x86_feature_detected!("sse2") {
                    unsafe { #sse2_name(#(#arg_names),*) }
                } else {
                    #scalar_name(#(#arg_names),*)
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                #[cfg(target_arch = "aarch64")]
                {
                    if ::std::arch::is_aarch64_feature_detected!("neon") {
                        unsafe { #neon_name(#(#arg_names),*) }
                    } else {
                        #scalar_name(#(#arg_names),*)
                    }
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    #scalar_name(#(#arg_names),*)
                }
            }
        }
    };

    TokenStream::from(expanded)
}
