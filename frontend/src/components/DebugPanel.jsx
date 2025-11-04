import { useState } from "react";

function DebugPanel({ response, visible }) {
  const debugLogs = response?.debug_logs;
  const sql = response?.sql;

  if (!visible || (!debugLogs && !sql)) {
    return null;
  }

  const [troubleshootingOpen, setTroubleshootingOpen] = useState(false);

  const openDebugWindow = () => {
    const debugWindow = window.open("", "_blank", "width=800,height=600");
    if (!debugWindow) {
      alert("Please allow pop-ups to view debug information.");
      return;
    }

    const htmlContent = `
      <!DOCTYPE html>
      <html>
        <head>
          <title>Debug Information</title>
          <style>
            body {
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
              margin: 0;
              padding: 20px;
              background: #f9fafb;
              color: #374151;
            }
            h1 {
              font-size: 20px;
              font-weight: 600;
              margin-bottom: 20px;
              color: #111827;
            }
            h2 {
              font-size: 14px;
              font-weight: 600;
              margin-top: 24px;
              margin-bottom: 8px;
              color: #4b5563;
            }
            pre {
              background: #1f2937;
              color: #f3f4f6;
              padding: 16px;
              border-radius: 8px;
              overflow-x: auto;
              font-size: 12px;
              font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
              line-height: 1.5;
            }
            .logs-pre {
              background: #f3f4f6;
              color: #1f2937;
              max-height: 400px;
              overflow-y: auto;
            }
          </style>
        </head>
        <body>
          <h1>Debug Information</h1>
          ${sql ? `
            <h2>Generated SQL Query</h2>
            <pre>${sql.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
          ` : ''}
          ${debugLogs ? `
            <h2>Debug Logs</h2>
            <pre class="logs-pre">${debugLogs.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</pre>
          ` : ''}
        </body>
      </html>
    `;

    debugWindow.document.write(htmlContent);
    debugWindow.document.close();
  };

  return (
    <section className="card p-6">
      <button
        type="button"
        onClick={() => setTroubleshootingOpen((prev) => !prev)}
        className="flex w-full items-center justify-between text-left text-sm font-semibold text-gray-800"
      >
        <span>🔧 Troubleshooting</span>
        <span className="text-xs text-gray-500">
          {troubleshootingOpen ? "Hide" : "Show"}
        </span>
      </button>
      {troubleshootingOpen && (
        <div className="mt-4 space-y-3 text-sm text-gray-700">
          <p className="text-xs text-gray-600">
            Having issues with query results? Click below to view technical details.
          </p>
          <button
            type="button"
            onClick={openDebugWindow}
            className="inline-flex items-center gap-2 rounded-md bg-indigo-600 px-4 py-2 text-xs font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            <span>📋</span>
            <span>View Debug Information</span>
            <span className="text-xs opacity-75">(opens in new window)</span>
          </button>
        </div>
      )}
    </section>
  );
}

export default DebugPanel;
