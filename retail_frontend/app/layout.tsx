import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "Nothierly",
  description: "Retail Gateway -- create an account and follow cached Nothierly performance.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="fr">
      <body style={{ fontFamily: "system-ui, sans-serif", margin: 0, background: "#0b0d12", color: "#e6e8eb" }}>
        <nav style={{ display: "flex", gap: "1.5rem", padding: "1rem 2rem", borderBottom: "1px solid #1f232b" }}>
          <a href="/" style={{ color: "inherit", textDecoration: "none", fontWeight: 700 }}>
            Nothierly
          </a>
          <a href="/login" style={{ color: "#9aa4b2", textDecoration: "none" }}>
            Connexion
          </a>
          <a href="/signup" style={{ color: "#9aa4b2", textDecoration: "none" }}>
            Créer un compte
          </a>
        </nav>
        <main style={{ maxWidth: 720, margin: "0 auto", padding: "2rem" }}>{children}</main>
      </body>
    </html>
  );
}
