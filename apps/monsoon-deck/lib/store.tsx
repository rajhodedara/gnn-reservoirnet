"use client";

/**
 * Lightweight app state via React context + useReducer. No Zustand dependency —
 * the state surface is small and entirely synchronous.
 *
 * The horizon scrubber is the killer interaction, so `horizon` (1..12) lives at the
 * root and every visualisation reads from it. Dragging it re-focuses the map, the
 * fan, the scoreboard and the level view together.
 */

import React, { createContext, useContext, useMemo, useReducer } from "react";
import type { AppData, EnsoLens, ReservoirId } from "./types";

export interface DeckState {
  data: AppData | null;
  loading: boolean;
  error: string | null;
  horizon: number;
  origin: string | null;
  selected: ReservoirId | null;
  hovered: ReservoirId | null;
  lens: EnsoLens;
  cascadeFrom: ReservoirId | null;
  /** Focus mode dims everything not related to the selection. */
  focusMode: boolean;
}

type Action =
  | { type: "data"; data: AppData }
  | { type: "error"; error: string }
  | { type: "loading" }
  | { type: "horizon"; value: number }
  | { type: "origin"; value: string }
  | { type: "select"; value: ReservoirId | null }
  | { type: "hover"; value: ReservoirId | null }
  | { type: "lens"; value: EnsoLens }
  | { type: "cascadeFrom"; value: ReservoirId | null }
  | { type: "focusMode"; value: boolean };

const initial: DeckState = {
  data: null,
  loading: true,
  error: null,
  horizon: 1,
  origin: null,
  selected: null,
  hovered: null,
  lens: "all",
  cascadeFrom: null,
  focusMode: true,
};

/** Seed the reducer from an already-fetched dataset so there is no empty first paint. */
function seeded(data: AppData | null): DeckState {
  if (!data) return initial;
  const origins = Object.keys(data.forecast).sort();
  return {
    ...initial,
    data,
    loading: false,
    origin: origins[origins.length - 1] ?? null,
    selected: data.reservoirs[0]?.id ?? null,
    cascadeFrom: data.cascade[0]?.upstream ?? null,
  };
}

function reducer(state: DeckState, action: Action): DeckState {
  switch (action.type) {
    case "loading":
      return { ...state, loading: true, error: null };
    case "data": {
      const origins = Object.keys(action.data.forecast).sort();
      return {
        ...state,
        data: action.data,
        loading: false,
        error: null,
        origin: state.origin && action.data.forecast[state.origin] ? state.origin : origins[origins.length - 1] ?? null,
        selected: state.selected ?? action.data.reservoirs[0]?.id ?? null,
        cascadeFrom: state.cascadeFrom ?? null,
      };
    }
    case "error":
      return { ...state, loading: false, error: action.error };
    case "horizon":
      return { ...state, horizon: Math.max(1, Math.min(12, Math.round(action.value))) };
    case "origin":
      return { ...state, origin: action.value };
    case "select":
      return { ...state, selected: action.value };
    case "hover":
      return { ...state, hovered: action.value };
    case "lens":
      return { ...state, lens: action.value };
    case "cascadeFrom":
      return { ...state, cascadeFrom: action.value };
    case "focusMode":
      return { ...state, focusMode: action.value };
    default:
      return state;
  }
}

interface Ctx {
  state: DeckState;
  dispatch: React.Dispatch<Action>;
  origins: string[];
}

const DeckCtx = createContext<Ctx | null>(null);

export function DeckProvider({
  children,
  initialData = null,
}: {
  children: React.ReactNode;
  initialData?: AppData | null;
}) {
  const [state, dispatch] = useReducer(reducer, initialData, seeded);
  const origins = useMemo(
    () => (state.data ? Object.keys(state.data.forecast).sort() : []),
    [state.data],
  );
  const value = useMemo(() => ({ state, dispatch, origins }), [state, origins]);
  return <DeckCtx.Provider value={value}>{children}</DeckCtx.Provider>;
}

export function useDeck(): Ctx {
  const ctx = useContext(DeckCtx);
  if (!ctx) throw new Error("useDeck must be used inside <DeckProvider>");
  return ctx;
}

/** Convenience hooks for the fields that are read most often. */
export function useHorizon(): number {
  return useDeck().state.horizon;
}

export function useSelected(): ReservoirId | null {
  return useDeck().state.selected;
}

/** The reservoir list, sorted the way the model reports them. */
export function useReservoirs() {
  return useDeck().state.data?.reservoirs ?? [];
}
