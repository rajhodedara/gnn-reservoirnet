"use client";

/**
 * Deck shell.
 *
 * Composition principle after the redesign: **one instrument, then its evidence.**
 *
 * The map occupies the focal position at ~2/3 width, paired with a single rail
 * that answers "what is this node telling me?" (identity + the signature fan).
 * Everything below the fold is *evidence* — each block carries a real sentence-case
 * heading and, where useful, the one question it answers. Long provenance prose is
 * demoted into an expandable note so it never competes with the numbers.
 */

import React, { useEffect, useState } from "react";
import type { AppData } from "@/lib/types";
import { loadAppData, loadGeo, type GeoAssets } from "@/lib/data";
import { DeckProvider, useDeck } from "@/lib/store";
import TopBar from "./TopBar";
import MapDeck from "./MapDeck";
import DetailPanel from "./DetailPanel";
import HorizonScrubber from "./HorizonScrubber";
import SkillBoard from "./SkillBoard";
import ClimateStrip from "./ClimateStrip";
import RobustnessView from "./RobustnessView";
import LevelView from "./LevelView";
import CascadeView from "./CascadeView";
import ProvenancePanel from "./ProvenancePanel";

export default function Deck() {
  const [data, setData] = useState<AppData | null>(null);
  const [geo, setGeo] = useState<GeoAssets>({ states: null, outline: null });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all([loadAppData(), loadGeo()])
      .then(([d, g]) => {
        if (!alive) return;
        setData(d);
        setGeo(g);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <main className="min-h-screen flex items-center justify-center p-8">
        <div className="panel ticks max-w-2xl p-6">
          <div className="eyebrow mb-2" style={{ color: "var(--rose)" }}>
            data unavailable
          </div>
          <p className="text-base text-data-text">{error}</p>
          <p className="note mt-3">
            This deck is a pure viewer over a single exported artifact. Nothing is hard-coded, so it
            cannot render without it.
          </p>
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="min-h-screen flex items-center justify-center">
        <span className="label animate-pulse">loading command deck…</span>
      </main>
    );
  }

  return (
    <DeckProvider initialData={data}>
      <Shell geo={geo} />
    </DeckProvider>
  );
}

/** A headed block of the deck. `q` states the question the block answers. */
function Block({
  title,
  q,
  children,
  right,
  className = "",
}: {
  title: string;
  q?: string;
  children: React.ReactNode;
  right?: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ticks ${className}`}>
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 px-4 py-3 border-b border-hair">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 min-w-0">
          <h2 className="sect whitespace-nowrap">{title}</h2>
          {q && <span className="label truncate">{q}</span>}
        </div>
        {right}
      </header>
      {children}
    </section>
  );
}

/** Collapsible long-form note. Keeps provenance available without shouting. */
function Note({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-hair">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full text-left px-4 py-2 label hover:text-data-dim flex items-center gap-2"
        aria-expanded={open}
      >
        <span className="text-data-faint">{open ? "▾" : "▸"}</span>
        {title}
      </button>
      {open && <div className="px-4 pb-3 note max-w-5xl">{children}</div>}
    </div>
  );
}

function Shell({ geo }: { geo: GeoAssets }) {
  const { state } = useDeck();
  if (!state.data) {
    return (
      <main className="min-h-screen flex items-center justify-center">
        <span className="label">initialising…</span>
      </main>
    );
  }
  const d = state.data;

  return (
    <div className="min-h-screen">
      <TopBar />

      <main className="px-4 py-4 space-y-4 max-w-[1760px] mx-auto">
        {/* ── The instrument ─────────────────────────────────────────────
            Map at focal weight; a single rail carries the selected node's
            identity and its signature fan. Nothing else competes up here. */}
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.85fr)_minmax(0,1fr)] gap-4 items-start">
          <section className="panel-hero ticks">
            <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 px-4 py-3 border-b border-hair">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 min-w-0">
                <h1 className="text-lg font-semibold tracking-[-0.01em] text-data-text">
                  Peninsular India reservoir network
                </h1>
                <span className="label">
                  {d.reservoirs.length} nodes · {d.cascade.length} hydraulic reaches · click a node to
                  focus
                </span>
              </div>
            </header>
            <div className="p-2.5">
              <MapDeck geo={geo} />
            </div>
            <Note title="How to read the map">
              Node ring = storage fill at the forecast origin, measured against that reservoir&apos;s
              gross capacity. The upward tick encodes the forecast inflow for the selected horizon on
              a log scale, so a dry node reads as an empty dashed ring rather than a small number.
              Solid teal reaches are the four hydraulic links the model actually propagates across;
              dotted blue links are climatological — rainfall correlation computed on the training
              period only. Basins are colour-coded, and the state outlines are real surveyed
              boundaries, not a stylised outline.
            </Note>
          </section>

          <DetailPanel />
        </div>

        {/* ── The driver ─────────────────────────────────────────────────
            The horizon scrubber governs the whole deck, so it gets its own
            full-width band rather than being buried in a grid. */}
        <HorizonScrubber />

        {/* ── Evidence ───────────────────────────────────────────────────
            Each block states the question it answers, so the reader knows
            what they are looking at without reading axis labels. */}

        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] gap-4 items-start">
          <Block
            title="Does the model actually beat the baselines?"
            q="Held-out skill per reservoir, losses included. Column NSE@wk tracks the selected horizon; the fixed metrics are the week-1 evaluation."
            right={
              <span className="eyebrow">
                {d.metrics.headline.n_seeds}-seed mean · {d.meta.test_year}
              </span>
            }
          >
            <SkillBoard />
          </Block>

          <Block title="Why is a given horizon uncertain?" q="climate teleconnections driving confidence">
            <ClimateStrip forecastIso={state.origin} />
          </Block>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 items-start">
          <Block title="Does week-1 skill survive other years?" q="rolling-origin 2020–2024">
            <RobustnessView />
          </Block>

          <Block title="How full are the reservoirs, and can we predict that?" q="storage vs capacity, and level skill">
            <LevelView />
          </Block>
        </div>

        <Block title="How does water travel down the cascade?" q="measured inflow lag per reach">
          <CascadeView />
        </Block>

        <Block title="Can this deck's numbers be trusted?" q="computed cross-checks and disclosed discrepancies">
          <ProvenancePanel />
        </Block>
      </main>

      <footer className="px-4 pb-10 max-w-[1760px] mx-auto">
        <div className="rule pt-4 flex flex-wrap gap-x-6 gap-y-2">
          <span className="eyebrow">ReservoirNet · Monsoon Command Deck</span>
          <span className="note">
            Every figure is read from <span className="num">app_data.json</span>, exported by{" "}
            <span className="num">scripts/export_app_data.py</span> from the repository&apos;s own model
            outputs and datasets. Nothing is hard-coded; unknown values render as an em dash and are
            never filled in.
          </span>
        </div>
      </footer>
    </div>
  );
}
