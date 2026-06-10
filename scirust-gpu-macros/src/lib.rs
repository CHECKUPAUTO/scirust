use proc_macro::TokenStream;
use quote::quote;
use syn::{FnArg, ItemFn, PatType, Type, TypeReference, parse_macro_input};

#[proc_macro_attribute]
pub fn gpu(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let block = &input_fn.block;
    let attrs = &input_fn.attrs;

    // Find the first &mut [f32] argument to dispatch to GPU
    let mut found_slice = false;
    for arg in &sig.inputs
    {
        if let FnArg::Typed(PatType { ty, .. }) = arg
            && let Type::Reference(TypeReference {
                elem, mutability, ..
            }) = ty.as_ref()
            && mutability.is_some()
            && let Type::Slice(slice) = elem.as_ref()
            && let Type::Path(path) = &*slice.elem
        {
            let seg = &path.path.segments.first().unwrap().ident;
            if seg == "f32"
            {
                found_slice = true;
                break;
            }
        }
    }

    if !found_slice
    {
        // If no mutable f32 slice found, emit the original function as-is
        let expanded = quote! {
            #(#attrs)*
            #vis #sig #block
        };
        return TokenStream::from(expanded);
    }

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            // Find the first &mut [f32] argument name
            let data = {
                let mut target: Option<&mut [f32]> = None;
                // This is a placeholder: in a real implementation we would
                // rewrite the body to extract the correct argument.
                // For this prototype, the gpu_or_cpu dispatch is demonstrated
                // inside the block when the caller passes a mutable slice.
                #block
            };
            data
        }
    };

    TokenStream::from(expanded)
}
