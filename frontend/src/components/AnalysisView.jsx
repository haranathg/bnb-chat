import { Suspense, lazy } from "react";
import DOMPurify from "dompurify";

const Plot = lazy(async () => {
  const [{ default: createPlotComponent }, plotlyModule] = await Promise.all([
    import("react-plotly.js/factory"),
    import("plotly.js/dist/plotly.min.js"),
  ]);
  const Plotly = plotlyModule.default ?? plotlyModule;
  return { default: createPlotComponent(Plotly) };
});

function AnalysisView({ response }) {
  const analysisHtml = response?.analysis_html
    ? DOMPurify.sanitize(response.analysis_html, {
        USE_PROFILES: { html: true },
      })
    : null;

  const charts = Array.isArray(response?.charts) ? response.charts : [];

  if (!response) {
    return (
      <div className="card p-6 text-sm text-gray-600">
        Insights will appear here after you run a query.
      </div>
    );
  }

  return (
    <section className="card space-y-6 p-6">
      <div className="space-y-4">
        <h2 className="text-base font-semibold text-gray-900">Analysis</h2>
        {analysisHtml ? (
          <article
            className="prose prose-sm max-w-none text-gray-800 prose-a:text-indigo-600 prose-img:rounded-lg"
            dangerouslySetInnerHTML={{ __html: analysisHtml }}
          />
        ) : (
          <p className="text-sm text-gray-500">No analysis provided.</p>
        )}
      </div>
      {charts.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-gray-800">Charts</h3>
          <div className="grid gap-6 md:grid-cols-2">
            {charts.map((figure, index) => (
              <div
                key={index}
                className="overflow-hidden rounded-xl border border-gray-200 bg-white p-2 shadow-sm"
              >
                <Suspense
                  fallback={
                    <div className="flex h-[360px] w-full items-center justify-center text-xs text-gray-500">
                      Loading chart…
                    </div>
                  }
                >
                  <Plot
                    data={figure.data || []}
                    layout={{
                      margin: { t: 40, r: 20, b: 40, l: 50 },
                      autosize: true,
                      height: 360,
                      paper_bgcolor: "transparent",
                      plot_bgcolor: "transparent",
                      template: "plotly_white",
                      font: {
                        family: "Inter, sans-serif",
                        size: 12,
                        color: "#1f2937",
                      },
                      legend: {
                        orientation: "h",
                        yanchor: "bottom",
                        y: -0.2,
                      },
                      ...(figure.layout || {}),
                    }}
                    config={{
                      responsive: true,
                      displaylogo: false,
                      modeBarButtonsToRemove: ["autoScale2d", "select2d", "lasso2d"],
                      toImageButtonOptions: {
                        format: "png",
                        filename: "bnb-chart",
                        scale: 2,
                      },
                      ...figure.config,
                    }}
                    style={{ width: "100%", height: "100%" }}
                    useResizeHandler
                    className="h-[360px] w-full"
                  />
                </Suspense>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

export default AnalysisView;
