//! The real backend: Tauri's IPC.
//!
//! Compiled only for `wasm32`, because it is the only place this code can
//! run — and keeping it out of host builds means the pure modules (the
//! reducers, the chart geometry, the wire decoding) are unit tested without
//! a browser or a WebAssembly toolchain.
//!
//! # How the call is made
//!
//! `window.__TAURI__.core.invoke` is looked up through reflection at call
//! time rather than bound as a `#[wasm_bindgen] extern` import. That is a
//! deliberate trade of a little ceremony for a much better failure: a page
//! opened outside the application gets
//! [`FrontendError::bridge_unavailable`] naming exactly what is missing,
//! instead of an uncatchable JavaScript `TypeError` during module
//! instantiation that leaves a blank window and nothing in the interface to
//! read.
//!
//! # Why the answer goes through JSON text
//!
//! The shell serialises its view types with `serde_json`; this side parses
//! them with `serde_json` built with `float_roundtrip`. Both directions of
//! that conversion are exact for `f64` — `ryu` emits the shortest text that
//! round-trips, and the parser reads it back bit for bit — so a coordinate
//! that reaches the chart is the coordinate the integrator produced. Since
//! results are validated to contain no non-finite values before they are
//! ever stored (`scirust_studio_runtime::validate_result`), the one shape
//! JSON cannot represent cannot arise.

use js_sys::{Function, JSON, Object, Promise, Reflect};
use serde::de::DeserializeOwned;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use super::wire::{
    BootstrapWire, CapabilityWire, ErrorWire, EventBatchWire, IntegrityWire, JobWire, RunWire,
    ScenarioWire, StoredRunWire, ValidationWire,
};
use super::{FrontendError, StudioBackend};

/// The real backend, talking to the Tauri shell over its IPC.
///
/// Zero-sized: it holds no connection, no cache and no credentials. Every
/// call is a fresh command invocation, so there is no stale state for the
/// interface to show after a worker restart.
#[derive(Debug, Clone, Copy, Default)]
pub struct TauriBridge;

/// Fetch `window.__TAURI__.core.invoke`.
///
/// Every failure names the exact property that was missing, because "the
/// backend is unavailable" without a reason is a bug report nobody can act
/// on.
fn invoke_function() -> Result<Function, FrontendError> {
    let window = web_sys::window()
        .ok_or_else(|| FrontendError::bridge_unavailable("there is no `window` object"))?;
    let tauri = Reflect::get(&window, &JsValue::from_str("__TAURI__"))
        .map_err(|_| FrontendError::bridge_unavailable("`window.__TAURI__` could not be read"))?;
    if tauri.is_undefined() || tauri.is_null()
    {
        return Err(FrontendError::bridge_unavailable(
            "`window.__TAURI__` is undefined",
        ));
    }
    let core = Reflect::get(&tauri, &JsValue::from_str("core"))
        .map_err(|_| FrontendError::bridge_unavailable("`__TAURI__.core` could not be read"))?;
    if core.is_undefined() || core.is_null()
    {
        return Err(FrontendError::bridge_unavailable(
            "`__TAURI__.core` is undefined",
        ));
    }
    let invoke = Reflect::get(&core, &JsValue::from_str("invoke")).map_err(|_| {
        FrontendError::bridge_unavailable("`__TAURI__.core.invoke` could not be read")
    })?;
    invoke
        .dyn_into::<Function>()
        .map_err(|_| FrontendError::bridge_unavailable("`__TAURI__.core.invoke` is not a function"))
}

/// Render a rejected value for a human.
fn describe(value: &JsValue) -> String {
    JSON::stringify(value)
        .ok()
        .and_then(|s| s.as_string())
        .or_else(|| value.as_string())
        .unwrap_or_else(|| "an unprintable value".to_string())
}

/// Call a command and decode its answer.
///
/// `args` is built by the caller as a plain JavaScript object. Tauri 2 maps
/// camelCase properties onto a command's snake_case parameters, which is why
/// the keys below are camelCase and why getting one wrong is a decode error
/// rather than a silent `None`.
async fn call<T: DeserializeOwned>(command: &str, args: Object) -> Result<T, FrontendError> {
    let invoke = invoke_function()?;
    let returned = invoke
        .call2(&JsValue::NULL, &JsValue::from_str(command), &args.into())
        .map_err(|e| FrontendError::opaque_rejection(command, describe(&e)))?;
    let promise = returned.dyn_into::<Promise>().map_err(|_| {
        FrontendError::bridge_unavailable("`__TAURI__.core.invoke` did not return a promise")
    })?;

    match JsFuture::from(promise).await
    {
        Ok(value) => decode(command, &value),
        Err(rejection) => Err(shell_error(command, &rejection)),
    }
}

/// Turn a resolved JavaScript value into a wire type.
fn decode<T: DeserializeOwned>(command: &str, value: &JsValue) -> Result<T, FrontendError> {
    let text = json_text(value)
        .ok_or_else(|| FrontendError::malformed_response(command, "the answer is not JSON"))?;
    serde_json::from_str(&text)
        .map_err(|e| FrontendError::malformed_response(command, e.to_string()))
}

/// The JSON text of a value, treating `undefined` as `null`.
fn json_text(value: &JsValue) -> Option<String> {
    if value.is_undefined()
    {
        return Some("null".to_string());
    }
    JSON::stringify(value).ok().and_then(|s| s.as_string())
}

/// Turn a rejection into an error card.
///
/// Every command in the shell rejects with a full error view, so the
/// structured path is the normal one; the opaque fallback exists for a
/// panic-in-a-command, which Tauri surfaces as a bare string.
fn shell_error(command: &str, rejection: &JsValue) -> FrontendError {
    match json_text(rejection).and_then(|t| serde_json::from_str::<ErrorWire>(&t).ok())
    {
        Some(wire) => FrontendError::from_wire(wire),
        None => FrontendError::opaque_rejection(command, describe(rejection)),
    }
}

/// A JavaScript object with one string property.
fn one_arg(key: &str, value: &str) -> Object {
    let object = Object::new();
    // `Reflect::set` on a fresh object cannot fail; the result is checked
    // anyway so a future change to `object` does not silently drop an
    // argument and produce a command call missing a parameter.
    let assigned =
        Reflect::set(&object, &JsValue::from_str(key), &JsValue::from_str(value)).unwrap_or(false);
    debug_assert!(assigned, "the argument object refused `{key}`");
    object
}

impl StudioBackend for TauriBridge {
    async fn bootstrap(&self) -> Result<BootstrapWire, FrontendError> {
        call("studio_bootstrap", Object::new()).await
    }

    async fn catalog(&self) -> Result<Vec<CapabilityWire>, FrontendError> {
        call("studio_catalog", Object::new()).await
    }

    async fn tutorials(&self) -> Result<Vec<CapabilityWire>, FrontendError> {
        call("studio_tutorials", Object::new()).await
    }

    async fn load_tutorial(&self, capability_id: &str) -> Result<ScenarioWire, FrontendError> {
        call(
            "studio_load_tutorial",
            one_arg("capabilityId", capability_id),
        )
        .await
    }

    async fn validate_scenario(&self, source: &str) -> Result<ValidationWire, FrontendError> {
        call("studio_validate_scenario", one_arg("source", source)).await
    }

    async fn start_run(&self, source: &str) -> Result<JobWire, FrontendError> {
        call("studio_start_run", one_arg("source", source)).await
    }

    async fn cancel_run(&self, job_id: &str) -> Result<(), FrontendError> {
        // The command returns `()`, which arrives as `null`.
        let _: Option<()> = call("studio_cancel_run", one_arg("jobId", job_id)).await?;
        Ok(())
    }

    async fn job_snapshot(&self, job_id: &str) -> Result<JobWire, FrontendError> {
        call("studio_job_snapshot", one_arg("jobId", job_id)).await
    }

    async fn active_job(&self) -> Result<Option<JobWire>, FrontendError> {
        call("studio_active_job", Object::new()).await
    }

    async fn list_runs(&self) -> Result<Vec<StoredRunWire>, FrontendError> {
        call("studio_list_runs", Object::new()).await
    }

    async fn load_run(&self, run_id: &str) -> Result<RunWire, FrontendError> {
        call("studio_load_run", one_arg("runId", run_id)).await
    }

    async fn verify_run(&self, run_id: &str) -> Result<IntegrityWire, FrontendError> {
        call("studio_verify_run", one_arg("runId", run_id)).await
    }

    async fn poll_events(&self) -> Result<EventBatchWire, FrontendError> {
        call("studio_poll_events", Object::new()).await
    }

    async fn store_path(&self) -> Result<String, FrontendError> {
        call("studio_store_path", Object::new()).await
    }

    async fn restart_worker(&self) -> Result<BootstrapWire, FrontendError> {
        call("studio_restart_worker", Object::new()).await
    }

    async fn worker_diagnostics(&self) -> Result<Vec<String>, FrontendError> {
        call("studio_worker_diagnostics", Object::new()).await
    }

    async fn frontend_ready(&self) -> Result<BootstrapWire, FrontendError> {
        call("frontend_ready", Object::new()).await
    }
}
