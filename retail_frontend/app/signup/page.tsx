"use client";

import { useState, type FormEvent } from "react";
import { ApiError, signup } from "@/lib/api";

export default function SignupPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState<"idle" | "submitting" | "done">("idle");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setStatus("submitting");
    setError(null);
    try {
      await signup(email, password);
      setStatus("done");
    } catch (err) {
      setStatus("idle");
      setError(err instanceof ApiError ? err.message : "Une erreur est survenue.");
    }
  }

  if (status === "done") {
    return (
      <section>
        <h1>Compte créé</h1>
        <p>
          Bienvenue -- votre compte a été créé. Un email de confirmation est en cours d&apos;envoi (traité en
          arrière-plan, la réponse ne l&apos;attend pas).
        </p>
      </section>
    );
  }

  return (
    <section>
      <h1>Créer un compte</h1>
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
          Mot de passe (12 caractères minimum)
          <input
            type="password"
            required
            minLength={12}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ display: "block", width: "100%", padding: "0.5rem", marginTop: "0.25rem" }}
          />
        </label>
        {error && <p style={{ color: "#e08080" }}>{error}</p>}
        <button type="submit" disabled={status === "submitting"} style={{ padding: "0.6rem", cursor: "pointer" }}>
          {status === "submitting" ? "Création..." : "Créer mon compte"}
        </button>
      </form>
    </section>
  );
}
