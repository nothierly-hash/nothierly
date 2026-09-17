"use client";

import { useState, type FormEvent } from "react";
import { ApiError, login } from "@/lib/api";

// Real, disclosed scope limit: this stores the access token in localStorage for this minimal
// entry point, not a full session-management layer (httpOnly cookie issuance would need a
// same-origin backend-for-frontend, real separate work -- see retail_gateway/main.py's own
// module docstring on the no-refresh-token scope boundary this pairs with).
export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<"idle" | "submitting" | "done">("idle");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setStatus("submitting");
    setError(null);
    try {
      const token = await login(email, password);
      window.localStorage.setItem("nothierly_access_token", token.access_token);
      setStatus("done");
    } catch (err) {
      setStatus("idle");
      setError(err instanceof ApiError ? err.message : "Une erreur est survenue.");
    }
  }

  if (status === "done") {
    return (
      <section>
        <h1>Connecté</h1>
        <p>Vous êtes maintenant authentifié.</p>
      </section>
    );
  }

  return (
    <section>
      <h1>Connexion</h1>
      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "0.75rem", maxWidth: 360 }}>
        <label>
          Email
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={{ display: "block", width: "100%", padding: "0.5rem", marginTop: "0.25rem" }}
          />
        </label>
        <label>
          Mot de passe
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ display: "block", width: "100%", padding: "0.5rem", marginTop: "0.25rem" }}
          />
        </label>
        {error && <p style={{ color: "#e08080" }}>{error}</p>}
        <button type="submit" disabled={status === "submitting"} style={{ padding: "0.6rem", cursor: "pointer" }}>
          {status === "submitting" ? "Connexion..." : "Se connecter"}
        </button>
      </form>
    </section>
  );
}
