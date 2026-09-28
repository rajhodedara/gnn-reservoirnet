"use client";

/**
 * Top bar: identity, instrument provenance, and the two global controls.
 *
 * Redesign changes: the two stacked rows of tiny uppercase mono are consolidated
 * into one identity row plus one quiet provenance row at the new type scale. The
 * origin's ENSO regime is stated in words next to a colour, so the amber/teal
 * accent is never the only carrier of meaning.
 */

import React from "react";
import type { EnsoLens } from "@/lib/types";
import { DASH } from "@/lib/format";
import { elNinoOnset, originRegimes } from "@/lib/selectors";
import { useDeck } from "@/lib/store";

const LENSES: Array<{ key: EnsoLens; label: string; tone: string }> = [
  { key: "all", label: "All", tone: "var(--txt)" },
  { key: "El Nino", label: "El Niño", tone: "var(--amber)" },
  { key: "Neutral", label: "Neutral", tone: "var(--teal)" },
];

export default function TopBar() {
  const { state, dispatch, origins } = useDeck();
  const data = state.data!;
  const onset = elNinoOnset(data);
  const regimes = originRegimes(data);

  const counts = {
    "El Nino": origins.filter((o) => regimes[o] === "El Nino").length,
    Neutral: origins.filter((o) => regimes[o] === "Neutral").length,
  };

  const originRegime = state.origin ? regimes[state.origin] : null;
  const regimeLabel =
    originRegime === "El Nino" ? "El Niño" : originRegime === "La Nina" ? "La Niña" : "Neutral";
  const regimeTone = originRegime === "El Nino" ? "var(--amber)" : "var(--teal)";

  return (
    <header className="border-b border-hair bg-ink-900/85 backdrop-blur-[3px] sticky top-0 z-20">
      <div className="px-4 py-2.5 flex flex-wrap items-center justify-between gap-x-6 gap-y-2 max-w-[1760px] mx-auto">
        {/* identity */}
        <div className="flex items-baseline gap-3">
          <span className="text-data-teal text-base leading-none" aria-hidden>
            ◆
          </span>
          <span className="text-base font-semibold tracking-[-0.01em] text-data-text">
            ReservoirNet
          </span>
          <span className="label">Monsoon Command Deck</span>
        </div>

        {/* controls */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          {/* origin + its regime, kept as one labelled unit so the regime word is not
              mistaken for one of the lens filter chips next to it */}
          <div className="flex items-baseline gap-2">
            <span className="eyebrow">Origin</span>
            <span className="num text-[12px] text-data-text">{state.origin ?? DASH}</span>
            {originRegime && (
              <span className="num text-[11.5px]" style={{ color: regimeTone }}>
                ({regimeLabel})
              </span>
            )}
          </div>

          <div className="flex items-baseline gap-2">
            <span className="eyebrow">2023 onset</span>
            <span className="num text-[12px]" style={{ color: "var(--amber)" }}>
              {onset.firstIso ? onset.firstIso.slice(0, 7) : DASH}
            </span>
          </div>

          {/* a real divider so the filter group reads as a separate control */}
          <span className="hidden sm:block w-px h-5" style={{ background: "var(--hair-2)" }} aria-hidden />

          <div className="flex items-center gap-1.5">
            <span className="eyebrow mr-0.5">Filter by regime</span>
            {LENSES.map((l) => {
              const on = state.lens === l.key;
              const n = l.key === "all" ? origins.length : counts[l.key as "El Nino" | "Neutral"] ?? 0;
              return (
                <button
                  key={l.key}
                  onClick={() => dispatch({ type: "lens", value: l.key })}
                  className="ctl ctl-sm num"
                  style={{
                    color: on ? l.tone : "var(--txt-3)",
                    borderColor: on ? `${l.tone}66` : "var(--hair)",
                    background: on ? `${l.tone}12` : "transparent",
                  }}
                  aria-pressed={on}
                  title={
                    l.key === "all"
                      ? "No regime filter"
                      : `${n} of ${origins.length} test origins classified as ${l.key} by the real monthly ONI`
                  }
                >
                  {l.label}
                  <span className="ml-1.5 opacity-70">{n}</span>
                </button>
              );
            })}
          </div>

          <button
            onClick={() => dispatch({ type: "focusMode", value: !state.focusMode })}
            className="ctl ctl-sm"
            data-on={state.focusMode ? "true" : "false"}
            aria-pressed={state.focusMode}
            title="Dim everything unrelated to the selected reservoir"
          >
            Focus {state.focusMode ? "on" : "off"}
          </button>
        </div>
      </div>

      {/* provenance strip — units and splits only. Model architecture detail lives
          in the About block at the foot of the deck, not in the operator's eyeline. */}
      <div className="px-4 pb-2 max-w-[1760px] mx-auto flex flex-wrap items-center gap-x-5 gap-y-1">
        <span className="label">
          train ≤ <span className="num text-data-text">{data.meta.train_end}</span> · val{" "}
          <span className="num text-data-text">{data.meta.val_year}</span> · test{" "}
          <span className="num text-data-text">{data.meta.test_year}</span>
        </span>
        <span className="label">
          inflow <span className="num text-data-text">{data.meta.units_inflow}</span>
        </span>
        <span className="label">
          storage <span className="num text-data-text">{data.meta.units_storage}</span>
        </span>
        <span className="label ml-auto">exported {data.meta.generated_at.replace("T", " ")}</span>
      </div>
    </header>
  );
}
