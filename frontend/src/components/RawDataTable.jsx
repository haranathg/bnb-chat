function RawDataTable({ response, visible }) {
  const rows = response?.raw_data;
  if (!visible || !rows || rows.length === 0) {
    return null;
  }

  let headers = [];
  let normalizedRows = [];

  if (Array.isArray(rows) && rows.length > 0) {
    if (Array.isArray(rows[0])) {
      headers =
        response?.raw_columns ||
        rows[0].map((_, index) => `Column ${index + 1}`);
      normalizedRows = rows;
    } else if (typeof rows[0] === "object") {
      headers = Object.keys(
        rows.reduce((acc, row) => Object.assign(acc, row), {})
      );
      normalizedRows = rows.map((row) => headers.map((header) => row[header]));
    }
  }

  return (
    <section className="card p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-base font-semibold text-gray-900">Raw data</h2>
        <span className="text-xs uppercase tracking-wide text-gray-400">
          Showing {normalizedRows.length} rows
        </span>
      </div>
      <div className="max-h-80 overflow-auto rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-100">
            <tr>
              {headers.map((header) => (
                <th
                  key={header}
                  scope="col"
                  className="px-4 py-2 text-left font-semibold uppercase tracking-wide text-gray-600"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 bg-white">
            {normalizedRows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className={rowIndex % 2 === 0 ? "bg-white" : "bg-gray-50"}
              >
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex} className="px-4 py-2 text-gray-700">
                    {cell === null || cell === undefined || cell === ""
                      ? "—"
                      : String(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default RawDataTable;
