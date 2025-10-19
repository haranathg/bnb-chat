import { useState } from "react";

function DebugPanel({ response, visible }) {
  const debugLogs = response?.debug_logs;
  const sql = response?.sql;

  if (!visible || (!debugLogs && !sql)) {
    return null;
  }

  const [open, setOpen] = useState(false);

  return (
    <section className="card p-6">
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex w-full items-center justify-between text-left text-sm font-semibold text-gray-800"
      >
        <span>Debug logs</span>
        <span className="text-xs text-gray-500">{open ? "Hide" : "Show"}</span>
      </button>
      {open && (
        <div className="mt-4 space-y-4 text-xs text-gray-700">
          {sql && (
            <div>
              <h3 className="font-semibold text-gray-600">SQL</h3>
              <pre className="mt-1 overflow-x-auto rounded-lg bg-gray-900 p-3 font-mono text-xs text-gray-100">
                {sql}
              </pre>
            </div>
          )}
          {debugLogs && (
            <div>
              <h3 className="font-semibold text-gray-600">Logs</h3>
              <pre className="mt-1 max-h-64 overflow-auto rounded-lg bg-gray-100 p-3 font-mono text-xs text-gray-800">
                {debugLogs}
              </pre>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

export default DebugPanel;
