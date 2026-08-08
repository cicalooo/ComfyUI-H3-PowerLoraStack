import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "H3PowerLoraStack";
const ROW_HEIGHT = 22;
const MARGIN = 12;

// Session cache for the loras folder listing.  Deliberately a module global
// rather than graph.extra: anything hung off the graph is serialized into the
// saved workflow, which would bake a stale copy of the whole folder into every
// .json.  The promise itself is cached so N rows opening at once share one
// request.
let loraListPromise = null;

/**
 * The stack node declares no ``lora_name`` input of its own -- rows are added
 * in the browser -- so the folder listing is borrowed from a node that does.
 */
function fetchLoraList() {
  if (!loraListPromise) {
    loraListPromise = api
      .fetchApi("/object_info/LoraLoaderModelOnly")
      .then((res) => res.json())
      .then(
        (info) => info?.LoraLoaderModelOnly?.input?.required?.lora_name?.[0] ?? []
      )
      .catch((err) => {
        console.warn("[H3PowerLoraStack] could not fetch lora list", err);
        loraListPromise = null;   // let the next picker retry
        return [];
      });
  }
  return loraListPromise;
}

function drawToggle(ctx, x, y, size, on) {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(x, y + 3, size * 1.7, size - 6, (size - 6) / 2);
  ctx.fillStyle = on ? "#3d8c5a" : "#3a3a3a";
  ctx.fill();
  ctx.beginPath();
  const knob = (size - 8) / 2;
  ctx.arc(x + (on ? size * 1.7 - knob - 3 : knob + 3), y + size / 2, knob, 0, Math.PI * 2);
  ctx.fillStyle = on ? "#d8f5e4" : "#8a8a8a";
  ctx.fill();
  ctx.restore();
}

function shortName(name) {
  if (!name || name === "None") return "click to choose";
  const base = String(name).replace(/\\/g, "/").split("/").pop();
  return base.replace(/\.(safetensors|ckpt|pt|bin)$/i, "");
}

/**
 * One stack row: enable toggle, lora name, strength, remove button.
 * Values serialize as {on, lora, strength}, which is what the Python side's
 * flexible `lora_N` inputs expect.
 */
function makeLoraWidget(node, name, value) {
  const widget = {
    name,
    type: "H3_LORA",
    value: Object.assign({ on: true, lora: "None", strength: 1.0 }, value || {}),
    options: { serialize: true },
    hitAreas: {},

    computeSize() {
      return [node.size[0], ROW_HEIGHT];
    },

    draw(ctx, node_, widgetWidth, y, height) {
      const margin = MARGIN;
      const left = margin;
      const right = widgetWidth - margin;
      const midY = y + height / 2;
      ctx.save();

      ctx.beginPath();
      ctx.roundRect(left, y + 1, right - left, height - 2, 6);
      ctx.fillStyle = this.value.on ? "#2b2b2b" : "#232323";
      ctx.fill();

      let cursor = left + 8;
      drawToggle(ctx, cursor, y, height, this.value.on);
      this.hitAreas.toggle = [cursor, cursor + height * 1.7];
      cursor += height * 1.7 + 10;

      // remove button, laid out from the right edge inward
      const removeX = right - 20;
      ctx.fillStyle = "#8a8a8a";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("✕", removeX + 6, midY);
      this.hitAreas.remove = [removeX, removeX + 16];

      const strengthRight = removeX - 10;
      const strengthLeft = strengthRight - 74;
      ctx.fillStyle = "#1e1e1e";
      ctx.beginPath();
      ctx.roundRect(strengthLeft, y + 3, strengthRight - strengthLeft, height - 6, 4);
      ctx.fill();
      ctx.fillStyle = "#b8b8b8";
      ctx.font = "11px sans-serif";
      ctx.fillText("◀", strengthLeft + 9, midY);
      ctx.fillText("▶", strengthRight - 9, midY);
      ctx.fillStyle = this.value.on ? "#e8e8e8" : "#777";
      ctx.font = "12px sans-serif";
      ctx.fillText(
        Number(this.value.strength).toFixed(2),
        (strengthLeft + strengthRight) / 2,
        midY
      );
      this.hitAreas.strengthDown = [strengthLeft, strengthLeft + 18];
      this.hitAreas.strengthUp = [strengthRight - 18, strengthRight];
      this.hitAreas.strengthValue = [strengthLeft + 18, strengthRight - 18];

      const nameLeft = cursor;
      const nameRight = strengthLeft - 10;
      ctx.textAlign = "left";
      ctx.fillStyle = this.value.on ? "#e8e8e8" : "#777";
      ctx.font = "12px sans-serif";
      const label = shortName(this.value.lora);
      let text = label;
      const maxWidth = nameRight - nameLeft;
      if (ctx.measureText(text).width > maxWidth) {
        while (text.length > 4 && ctx.measureText(text + "…").width > maxWidth) {
          text = text.slice(0, -1);
        }
        text += "…";
      }
      ctx.fillText(text, nameLeft, midY);
      this.hitAreas.name = [nameLeft, nameRight];

      ctx.restore();
    },

    mouse(event, pos, node_) {
      if (event.type !== "pointerdown") return false;
      const x = pos[0];
      const hit = (area) => area && x >= area[0] && x <= area[1];

      if (hit(this.hitAreas.toggle)) {
        this.value = { ...this.value, on: !this.value.on };
        node_.setDirtyCanvas(true, true);
        return true;
      }
      if (hit(this.hitAreas.remove)) {
        removeLoraWidget(node_, this);
        return true;
      }
      if (hit(this.hitAreas.strengthDown)) {
        this.value = { ...this.value, strength: round2(this.value.strength - 0.05) };
        node_.setDirtyCanvas(true, true);
        return true;
      }
      if (hit(this.hitAreas.strengthUp)) {
        this.value = { ...this.value, strength: round2(this.value.strength + 0.05) };
        node_.setDirtyCanvas(true, true);
        return true;
      }
      if (hit(this.hitAreas.strengthValue)) {
        app.canvas.prompt(
          "Strength",
          this.value.strength,
          (v) => {
            const parsed = parseFloat(v);
            if (!Number.isNaN(parsed)) {
              this.value = { ...this.value, strength: parsed };
              node_.setDirtyCanvas(true, true);
            }
          },
          event
        );
        return true;
      }
      if (hit(this.hitAreas.name)) {
        showLoraMenu(node_, this, event);
        return true;
      }
      return false;
    },

    serializeValue() {
      return { ...this.value };
    },
  };
  return widget;
}

function round2(v) {
  return Math.round(v * 100) / 100;
}

/* ---------------------------------------------------------------- picker -- */

const PICKER_CSS = `
.h3pls-picker{position:fixed;z-index:10000;display:flex;flex-direction:column;
  width:460px;max-width:92vw;background:#1e1e1e;border:1px solid #4a4a4a;
  border-radius:8px;box-shadow:0 12px 32px rgba(0,0,0,.6);overflow:hidden;
  font-family:system-ui,-apple-system,sans-serif;}
.h3pls-picker input{margin:8px;padding:7px 9px;background:#111;color:#eee;
  border:1px solid #555;border-radius:5px;font-size:13px;outline:none;}
.h3pls-picker input:focus{border-color:#3d8c5a;}
.h3pls-count{padding:0 12px 6px;font-size:11px;color:#777;}
.h3pls-list{overflow-y:auto;max-height:46vh;}
.h3pls-item{padding:6px 12px;font-size:12.5px;color:#ddd;cursor:pointer;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.h3pls-item:hover{background:#2c2c2c;}
.h3pls-item.sel{background:#31543f;}
.h3pls-item .dir{color:#7a7a7a;}
.h3pls-item .hit{color:#7fd6a0;font-weight:600;}
.h3pls-item.cur::after{content:"current";float:right;color:#666;font-size:10px;}
.h3pls-empty{padding:14px 12px;font-size:12px;color:#888;}
`;

function ensurePickerCss() {
  if (document.getElementById("h3pls-css")) return;
  const style = document.createElement("style");
  style.id = "h3pls-css";
  style.textContent = PICKER_CSS;
  document.head.appendChild(style);
}

/** Case-insensitive AND-match of every whitespace-separated term. */
function matchTerms(name, terms) {
  const hay = name.toLowerCase();
  const spans = [];
  for (const term of terms) {
    const at = hay.indexOf(term);
    if (at < 0) return null;
    spans.push([at, at + term.length]);
  }
  return spans;
}

/** Basename hits and early hits rank first; then shortest name. */
function score(name, terms, spans) {
  if (!terms.length) return 0;
  const slash = name.lastIndexOf("/") + 1;
  let s = 0;
  for (const [start] of spans) {
    if (start >= slash) s -= 1000;
    s += start;
  }
  return s + name.length * 0.01;
}

function highlight(name, spans) {
  const slash = name.lastIndexOf("/") + 1;
  const merged = [];
  for (const [a, b] of [...spans].sort((p, q) => p[0] - q[0])) {
    const last = merged[merged.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else merged.push([a, b]);
  }
  const esc = (t) =>
    t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let html = "";
  let at = 0;
  for (const [a, b] of merged) {
    html += wrapDir(esc(name.slice(at, a)), at, slash);
    html += `<span class="hit">${esc(name.slice(a, b))}</span>`;
    at = b;
  }
  html += wrapDir(esc(name.slice(at)), at, slash);
  return html;
}

function wrapDir(text, offset, slash) {
  if (!text) return "";
  if (offset + text.length <= slash) return `<span class="dir">${text}</span>`;
  if (offset >= slash) return text;
  const cut = slash - offset;
  return `<span class="dir">${text.slice(0, cut)}</span>${text.slice(cut)}`;
}

/**
 * Filterable LoRA picker. LiteGraph's ContextMenu has no search, which is
 * unusable once the loras folder holds more than a screenful, so this is a
 * self-contained overlay: type to narrow, arrows to move, Enter to take it.
 */
async function showLoraMenu(node, widget, event, { removeOnCancel = false } = {}) {
  const names = await fetchLoraList();
  if (!names.length) {
    if (removeOnCancel) removeLoraWidget(node, widget);
    alert("No LoRAs found in models/loras.");
    return;
  }
  ensurePickerCss();

  const current = widget.value.lora;
  const root = document.createElement("div");
  root.className = "h3pls-picker";
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Type to filter…  (space-separated terms all must match)";
  input.spellcheck = false;
  const count = document.createElement("div");
  count.className = "h3pls-count";
  const list = document.createElement("div");
  list.className = "h3pls-list";
  root.append(input, count, list);
  document.body.appendChild(root);

  // place near the click, clamped into the viewport
  const px = event?.clientX ?? window.innerWidth / 2;
  const py = event?.clientY ?? window.innerHeight / 2;
  root.style.left = `${Math.min(px, window.innerWidth - 470)}px`;
  root.style.top = `${Math.min(py, window.innerHeight - 320)}px`;

  let rows = [];
  let sel = 0;

  let committed = false;
  const close = () => {
    document.removeEventListener("pointerdown", onOutside, true);
    root.remove();
    // a row opened straight from "Add LoRA" and then dismissed was never
    // wanted, so take it back out rather than leaving an empty slot behind
    if (removeOnCancel && !committed) removeLoraWidget(node, widget);
  };
  const commit = (value) => {
    committed = true;
    widget.value = { ...widget.value, lora: value };
    node.setDirtyCanvas(true, true);
    close();
  };
  const onOutside = (e) => {
    if (!root.contains(e.target)) close();
  };

  const render = () => {
    const query = input.value.trim().toLowerCase();
    const terms = query ? query.split(/\s+/) : [];
    rows = [];
    for (const name of names) {
      const spans = matchTerms(name, terms);
      if (spans === null) continue;
      rows.push({ name, spans, s: score(name, terms, spans) });
    }
    if (terms.length) rows.sort((a, b) => a.s - b.s);
    sel = 0;
    if (!terms.length) {
      const at = rows.findIndex((r) => r.name === current);
      if (at > 0) sel = at;
    }

    count.textContent = `${rows.length} of ${names.length}`;
    list.replaceChildren();
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "h3pls-empty";
      empty.textContent = "No LoRA matches that filter.";
      list.appendChild(empty);
      return;
    }
    rows.forEach((row, i) => {
      const item = document.createElement("div");
      item.className =
        "h3pls-item" + (i === sel ? " sel" : "") + (row.name === current ? " cur" : "");
      item.innerHTML = highlight(row.name, row.spans);
      item.title = row.name;
      item.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        e.stopPropagation();
        commit(row.name);
      });
      list.appendChild(item);
    });
    scrollToSel();
  };

  const scrollToSel = () => {
    const el = list.children[sel];
    if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest" });
  };

  const move = (delta) => {
    if (!rows.length) return;
    const prev = list.children[sel];
    if (prev) prev.classList.remove("sel");
    sel = (sel + delta + rows.length) % rows.length;
    const next = list.children[sel];
    if (next) next.classList.add("sel");
    scrollToSel();
  };

  input.addEventListener("input", render);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "PageDown") { e.preventDefault(); move(10); }
    else if (e.key === "PageUp") { e.preventDefault(); move(-10); }
    else if (e.key === "Enter") {
      e.preventDefault();
      if (rows[sel]) commit(rows[sel].name);
    } else if (e.key === "Escape") {
      e.preventDefault();
      close();
    }
    e.stopPropagation();   // keep ComfyUI's global hotkeys out of the box
  });

  render();
  const grabFocus = () => {
    if (root.isConnected && document.activeElement !== input) input.focus();
  };
  // Defer by a full task, not a microtask: the pointerdown that opened this is
  // still being dispatched, and a capture listener added now would see it and
  // close the picker immediately.
  setTimeout(() => {
    document.addEventListener("pointerdown", onOutside, true);
    grabFocus();
  }, 0);
  // LiteGraph pulls focus back to the canvas on a delay in some paths; re-assert
  // once so the caret really is in the filter box and typing goes there.
  setTimeout(grabFocus, 60);
}

function loraWidgets(node) {
  return (node.widgets || []).filter((w) => w.type === "H3_LORA");
}

function renumber(node) {
  loraWidgets(node).forEach((w, i) => {
    w.name = `lora_${i + 1}`;
  });
}

function addLoraWidget(node, value) {
  const widget = makeLoraWidget(node, `lora_${loraWidgets(node).length + 1}`, value);
  node.widgets = node.widgets || [];
  // keep the "Add LoRA" button last
  const buttonIndex = node.widgets.findIndex((w) => w.name === "➕ Add LoRA");
  if (buttonIndex >= 0) node.widgets.splice(buttonIndex, 0, widget);
  else node.widgets.push(widget);
  renumber(node);
  resize(node);
  return widget;
}

function removeLoraWidget(node, widget) {
  const i = node.widgets.indexOf(widget);
  if (i >= 0) node.widgets.splice(i, 1);
  renumber(node);
  resize(node);
  node.setDirtyCanvas(true, true);
}

function resize(node) {
  const [minWidth, minHeight] = node.computeSize();
  // Width stays the user's choice; height tracks the row count so removing a
  // row hands the space back instead of leaving a hole.
  node.setSize([Math.max(node.size[0], minWidth), minHeight]);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "h3.power.lora.stack",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      this.serialize_widgets = true;

      // (value, canvas, node, pos, event) -- the event is the pointerdown that
      // started the click, so the picker opens right under the button.
      this.addWidget("button", "➕ Add LoRA", null, (_v, _canvas, _node, _pos, event) => {
        const widget = addLoraWidget(this);
        // open the picker immediately: the row is added to be filled in, so the
        // filter box takes the keyboard straight away
        showLoraMenu(this, widget, event, { removeOnCancel: true });
      });

      fetchLoraList();   // warm the cache so the first picker opens instantly
      resize(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      onConfigure?.apply(this, arguments);
      // ComfyUI restores widgets_values positionally against the widgets that
      // existed before this ran, which is only the fixed combos and the button.
      // The stack rows are recovered by shape instead of position, so a slot
      // shifting cannot corrupt them.
      const values = info?.widgets_values;
      if (!Array.isArray(values)) return;
      const rows = values.filter(
        (v) => v && typeof v === "object" && !Array.isArray(v) && "lora" in v
      );
      for (const w of loraWidgets(this)) removeLoraWidget(this, w);
      for (const row of rows) addLoraWidget(this, row);
      resize(this);
    };
  },
});
