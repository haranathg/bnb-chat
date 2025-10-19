import { useState } from "react";

function Login({ onLogin }) {
  const [tokenInput, setTokenInput] = useState("");
  const [error, setError] = useState(null);

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!tokenInput.trim()) {
      setError("Token is required.");
      return;
    }
    setError(null);
    onLogin(tokenInput.trim());
  };

  return (
    <div className="w-full max-w-md rounded-2xl border border-gray-200 bg-white p-8 shadow-lg">
      <h2 className="mb-4 text-xl font-semibold text-gray-900">
        Enter access token
      </h2>
      <p className="mb-6 text-sm text-gray-600">
        Provide the shared token to continue using bnb-chat-with-data.
      </p>
      <form onSubmit={handleSubmit} className="space-y-4">
        <label className="block text-sm font-medium text-gray-700">
          Access token
          <input
            type="password"
            className="mt-1 w-full rounded-lg border-gray-300 text-sm shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            placeholder="bnb-secret-2025"
            value={tokenInput}
            onChange={(event) => setTokenInput(event.target.value)}
          />
        </label>
        {error && <p className="text-sm text-red-600">{error}</p>}
        <button
          type="submit"
          className="w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
        >
          Continue
        </button>
      </form>
    </div>
  );
}

export default Login;
