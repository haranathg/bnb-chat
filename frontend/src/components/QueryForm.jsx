function QueryForm({
  query,
  onChange,
  onSubmit,
  loading,
  showRaw,
  showDebug,
  onToggleRaw,
  onToggleDebug,
}) {
  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit();
  };

  return (
    <form onSubmit={handleSubmit} className="card p-6">
      <div className="flex flex-col gap-4">
        <label className="block text-sm font-medium text-gray-700">
          Ask a data question
          <textarea
            className="mt-2 min-h-[120px] w-full rounded-lg border-gray-300 text-sm leading-relaxed shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            placeholder="Which drug classes have shown the highest variance between AWP and ASP?"
            value={query}
            onChange={(event) => onChange(event.target.value)}
          />
        </label>
        <div className="flex flex-wrap items-center gap-6 text-sm text-gray-700">
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
              checked={showRaw}
              onChange={(event) => onToggleRaw(event.target.checked)}
            />
            Show raw data
          </label>
          <label className="inline-flex items-center gap-2">
            <input
              type="checkbox"
              className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
              checked={showDebug}
              onChange={(event) => onToggleDebug(event.target.checked)}
            />
            Show debug logs
          </label>
        </div>
        <div>
          <button
            type="submit"
            disabled={loading}
            className="inline-flex items-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:bg-indigo-400"
          >
            {loading ? "Running query…" : "Run Query"}
          </button>
        </div>
      </div>
    </form>
  );
}

export default QueryForm;
