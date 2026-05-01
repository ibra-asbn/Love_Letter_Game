export function readCurrentView() {
  return new URLSearchParams(window.location.search).get("view");
}

export function pushView(view) {
  const url = new URL(window.location.href);
  if (view) {
    url.searchParams.set("view", view);
  } else {
    url.searchParams.delete("view");
  }
  window.history.pushState({}, "", url);
  window.dispatchEvent(new Event("palace:viewchange"));
}
