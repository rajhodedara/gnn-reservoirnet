import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ReservoirNet · Monsoon Command Deck",
  description:
    "Next 1–12 week inflow and storage forecasts for 10 major reservoirs of Peninsular India — spatio-temporal GNN, quantile bands, and honest skill against persistence and climatology.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
