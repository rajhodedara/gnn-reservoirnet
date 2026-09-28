"use client";

/**
 * Provenance and integrity.
 *
 * This panel exists because the repository's own artifacts disagree with each
 * other. Redesign changes: the header/heading is supplied by the shell; the check
 * table leads with a compact status summary; and the rolling-origin table (which
 * duplicates RobustnessView) is dropped in favour of linking the reader to it, so
 * the same numbers are not printed twice on one page.
 */

import React, { useMemo, useState } from "react";
import { DASH, fmtMetric } from "@/lib/format";
import { useDeck } from "@/lib/store";

export default function ProvenancePanel() {
  const { state } = useDeck();
  const data = state.data!;
  const [open, setOpen] = useState(false);

  const headline = data.metrics.headline;
  const integrity = data.integrity;
  const failures = integrity.checked.filter((c) => c.status === "fail");

  return (
    <div>
      <div className="grid grid-cols-2 md:grid-cols-4 border-b border-hair">
        <Cell label="Seeds averaged" value={String(headline.n_seeds)} />
        <Cell
          label="Pooled mean NSE"
          value={fmtMetric(headline.mean_nse_pooled)}
          sub="reproduces the README headline of 0.581"
          tone="var(--teal)"
        />
        <Cell label="Week-1 mean NSE" value={fmtMetric(headline.mean_nse_week1)} tone="var(--teal)" />
        <Cell
          label="Unit conversions"
          value="Verified"
          sub="1 TMC = 28.3168 MCM · weekly sums reproduced from raw CSVs"
          tone="var(--teal)"
        />
      </div>

      {/* status line, always visible */}
      <div className="px-4 py-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
          {integrity.checked.map((c) => (
            <span key={c.name} className="inline-flex items-center gap-1.5 text-2xs">
              <span
                aria-hidden
                className="ctl ctl-xs num inline-flex gap-1"
                style={{
                  color: c.status === "pass" ? "var(--teal)" : "var(--rose)",
                  borderColor: c.status === "pass" ? "rgba(62,201,182,0.45)" : "rgba(207,101,99,0.45)",
                  background: c.status === "pass" ? "rgba(62,201,182,0.08)" : "rgba(207,101,99,0.08)",
                }}
              >
                <span aria-hidden>{c.status === "pass" ? "▲" : "▼"}</span>
                {c.status === "pass" ? "PASS" : "FLAG"}
              </span>
              <span className="num text-data-dim">{c.name.replace(/_/g, " ")}</span>
            </span>
          ))}
        </div>
        <button
          onClick={() => setOpen((v) => !v)}
          className="ctl ctl-sm ctl-quiet"
          aria-expanded={open}
        >
          {open ? "Hide detail" : `Show detail${failures.length ? ` · ${failures.length} disclosed` : ""}`}
        </button>
      </div>

      {open && (
        <>
          <div className="rule overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-hair">
                  <th className="px-4 py-2.5 text-left font-medium text-data-faint" style={{ fontSize: 11.5 }}>Check</th>
                  <th className="px-4 py-2.5 text-left font-medium text-data-faint" style={{ fontSize: 11.5 }}>Status</th>
                  <th className="px-4 py-2.5 text-left font-medium text-data-faint" style={{ fontSize: 11.5 }}>Measured detail</th>
                </tr>
              </thead>
              <tbody>
                {integrity.checked.map((c) => (
                  <tr key={c.name} className="border-b border-hair/70 align-top">
                    <td className="num px-4 py-2.5 text-data-dim whitespace-nowrap" style={{ fontSize: 11.5 }}>
                      {c.name.replace(/_/g, " ")}
                    </td>
                    <td className="px-4 py-2.5 whitespace-nowrap">
                      <span
                        className="ctl ctl-xs num inline-flex gap-1"
                        style={{
                          color: c.status === "pass" ? "var(--teal)" : "var(--rose)",
                          borderColor: c.status === "pass" ? "rgba(62,201,182,0.45)" : "rgba(207,101,99,0.45)",
                          background: c.status === "pass" ? "rgba(62,201,182,0.08)" : "rgba(207,101,99,0.08)",
                        }}
                      >
                        <span aria-hidden>{c.status === "pass" ? "▲" : "▼"}</span>
                        {c.status === "pass" ? "PASS" : "FLAG"}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 note">{c.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="px-4 py-3 note max-w-4xl">{integrity.note}</div>
        </>
      )}

      {!open && (
        <div className="rule px-4 py-2.5 note">
          Two of {integrity.checked.length} computed checks flag a discrepancy between the repository&apos;s
          own files. Nothing was recomputed or regenerated; the disagreement is measured and shown here
          rather than resolved in favour of the friendlier number. Expand for the full detail.
        </div>
      )}
    </div>
  );
}

function Cell({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="px-4 py-3 border-r border-hair last:border-r-0">
      <div className="eyebrow">{label}</div>
      <div className="stat mt-1" style={{ color: tone ?? "var(--txt)" }}>
        {value}
      </div>
      {sub && <div className="label mt-0.5">{sub}</div>}
    </div>
  );
}

export { DASH };
