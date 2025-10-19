import { useEffect, useMemo, useState } from "react";
import { joinPaths } from "./utils/path";
import Login from "./components/Login.jsx";
import Sidebar from "./components/Sidebar.jsx";
import QueryForm from "./components/QueryForm.jsx";
import AnalysisView from "./components/AnalysisView.jsx";
import RawDataTable from "./components/RawDataTable.jsx";
import DebugPanel from "./components/DebugPanel.jsx";

const TOKEN_KEY = "bnb_token";
const HISTORY_KEY = "bnb_history";
const CACHE_KEY = "bnb_cache";
const CACHE_LIMIT = 5;
const HISTORY_LIMIT = 20;
const CACHE_TTL = 10 * 60 * 1000; // 10 minutes

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

const getInitialHistory = () => {
  try {
    const stored = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(stored) ? stored.slice(0, HISTORY_LIMIT) : [];
  } catch {
    return [];
  }
};

const getInitialCache = () => {
  try {
    const raw = JSON.parse(localStorage.getItem(CACHE_KEY) || "{}");
    if (!raw || typeof raw !== "object") return {};
    const now = Date.now();
    return Object.fromEntries(
      Object.entries(raw)
        .filter(([, value]) => now - value.timestamp < CACHE_TTL)
        .sort((a, b) => b[1].timestamp - a[1].timestamp)
        .slice(0, CACHE_LIMIT)
    );
  } catch {
    return {};
  }
};

function App() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [history, setHistory] = useState(getInitialHistory);
  const [cache, setCache] = useState(getInitialCache);
  const [showRaw, setShowRaw] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }, [history]);

  useEffect(() => {
    localStorage.setItem(CACHE_KEY, JSON.stringify(cache));
  }, [cache]);

  const addToHistory = (nextQuery) => {
    setHistory((prev) => {
      const updated = [nextQuery, ...prev.filter((item) => item !== nextQuery)];
      return updated.slice(0, HISTORY_LIMIT);
    });
  };

  const updateCache = (nextQuery, data) => {
    setCache((prev) => {
      const merged = {
        ...prev,
        [nextQuery]: { data, timestamp: Date.now() },
      };
      const trimmedEntries = Object.entries(merged)
        .sort((a, b) => b[1].timestamp - a[1].timestamp)
        .slice(0, CACHE_LIMIT);
      return Object.fromEntries(trimmedEntries);
    });
  };

  const loadFromCache = (targetQuery) => {
    const cachedItem = cache[targetQuery];
    if (!cachedItem) return false;
    if (Date.now() - cachedItem.timestamp > CACHE_TTL) {
      setCache((prev) => {
        const cloned = { ...prev };
        delete cloned[targetQuery];
        return cloned;
      });
      return false;
    }
    setResponse(cachedItem.data);
    setError(null);
    return true;
  };

  const handleLogout = () => {
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setResponse(null);
  };

  const runQuery = async (targetQuery, opts = {}) => {
    if (!targetQuery) {
      setError("Please provide a question before running the query.");
      return;
    }
    setError(null);

    if (loadFromCache(targetQuery)) {
      addToHistory(targetQuery);
      return;
    }

    setIsLoading(true);
    try {
      const res = await fetch(joinPaths(API_BASE_URL, "/query"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${localStorage.getItem(TOKEN_KEY) || ""}`,
        },
        body: JSON.stringify({
          query: targetQuery,
          options: {
            show_raw: opts.showRaw ?? showRaw,
            debug: opts.showDebug ?? showDebug,
          },
        }),
      });

      if (res.status === 401) {
        handleLogout();
        setError("Session expired. Please enter the access token again.");
        return;
      }

      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || "Unexpected error from server.");
      }

      const data = await res.json();
      setResponse(data);
      updateCache(targetQuery, data);
      addToHistory(targetQuery);
    } catch (err) {
      setError(err.message || "Unable to process query.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = () => {
    runQuery(query);
  };

  const handleHistorySelect = (selectedQuery) => {
    setQuery(selectedQuery);
    if (!loadFromCache(selectedQuery)) {
      runQuery(selectedQuery);
    }
    setSidebarOpen(false);
  };

  const handleLogin = (nextToken) => {
    localStorage.setItem(TOKEN_KEY, nextToken);
    setToken(nextToken);
  };

  const mainContent = useMemo(
    () => (
      <div className="flex flex-col gap-6">
        <QueryForm
          query={query}
          onChange={setQuery}
          onSubmit={handleSubmit}
          loading={isLoading}
          showRaw={showRaw}
          showDebug={showDebug}
          onToggleRaw={setShowRaw}
          onToggleDebug={setShowDebug}
        />
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}
        <AnalysisView response={response} />
        <RawDataTable response={response} visible={showRaw} />
        <DebugPanel
          response={response}
          visible={showDebug}
        />
      </div>
    ),
    [query, showRaw, showDebug, response, isLoading, error]
  );

  if (!token) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-100 px-4">
        <Login onLogin={handleLogin} />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-gray-50">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        history={history}
        onSelect={handleHistorySelect}
        onClear={() => {
          setHistory([]);
          setCache({});
        }}
      />
      <div className="flex flex-1 flex-col">
        <header className="sticky top-0 z-10 border-b border-gray-200 bg-white/80 backdrop-blur">
          <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-4">
            <div className="flex items-center gap-3">
              <button
                className="inline-flex items-center rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 lg:hidden"
                type="button"
                onClick={() => setSidebarOpen(true)}
              >
                <span className="sr-only">Open history</span>
                &#9776;
              </button>
              <h1 className="text-lg font-semibold text-gray-900">
                bnb-chat-with-data
              </h1>
            </div>
            <button
              onClick={handleLogout}
              className="text-sm font-medium text-gray-500 hover:text-gray-700"
            >
              Log out
            </button>
          </div>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
          {mainContent}
        </main>
      </div>
    </div>
  );
}

export default App;
