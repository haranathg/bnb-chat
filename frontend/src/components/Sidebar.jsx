import { Fragment } from "react";

function Sidebar({ open, onClose, history, onSelect, onClear }) {
  const content = (
    <div className="flex h-full w-64 flex-col border-r border-gray-200 bg-white">
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-4">
        <h2 className="text-sm font-semibold text-gray-800">Recent questions</h2>
        <button
          type="button"
          className="text-xs font-medium text-indigo-600 hover:text-indigo-700"
          onClick={onClear}
        >
          Clear
        </button>
      </div>
      <nav className="flex-1 overflow-y-auto px-2 py-3">
        {history.length === 0 ? (
          <p className="px-2 text-sm text-gray-500">
            Your latest questions will appear here.
          </p>
        ) : (
          <ul className="space-y-1">
            {history.slice(0, 20).map((item) => (
              <li key={item}>
                <button
                  type="button"
                  onClick={() => onSelect(item)}
                  className="w-full rounded-lg px-3 py-2 text-left text-sm text-gray-700 hover:bg-indigo-50 hover:text-indigo-700"
                >
                  {item.length > 80 ? `${item.slice(0, 77)}…` : item}
                </button>
              </li>
            ))}
          </ul>
        )}
      </nav>
    </div>
  );

  return (
    <Fragment>
      <div className="hidden lg:block">{content}</div>
      {open && (
        <div className="fixed inset-0 z-40 flex lg:hidden">
          <div className="absolute inset-0 bg-gray-900/50" onClick={onClose} />
          <div className="relative ml-auto h-full w-72 shadow-xl">{content}</div>
        </div>
      )}
    </Fragment>
  );
}

export default Sidebar;
