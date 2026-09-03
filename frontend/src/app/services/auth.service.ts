import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';

export interface AuthResponse {
  token: string;
  username: string;
  expires_at: string;
  is_admin: boolean;
}

interface StoredSession {
  token: string;
  username: string;
  expires_at: string;
  isAdmin: boolean;
}

const STORAGE_KEY = 'document-analyzer.session';

@Injectable({ providedIn: 'root' })
export class AuthService {
  // See api.service.ts - relative so the same build works in dev and prod.
  private readonly baseUrl = 'api/auth';
  private session: StoredSession | null = null;

  /** Emits the signed-in username, or null when signed out. */
  readonly user$ = new BehaviorSubject<string | null>(null);

  constructor(private http: HttpClient) {
    this.session = this.restore();
    this.user$.next(this.session?.username ?? null);
  }

  get token(): string | null {
    return this.session?.token ?? null;
  }

  get username(): string | null {
    return this.session?.username ?? null;
  }

  get isSignedIn(): boolean {
    return this.session !== null;
  }

  get isAdmin(): boolean {
    return this.session?.isAdmin ?? false;
  }

  login(username: string, password: string): Observable<AuthResponse> {
    return this.http
      .post<AuthResponse>(`${this.baseUrl}/login`, { username, password })
      .pipe(tap((res) => this.store(res)));
  }

  register(username: string, password: string): Observable<AuthResponse> {
    return this.http
      .post<AuthResponse>(`${this.baseUrl}/register`, { username, password })
      .pipe(tap((res) => this.store(res)));
  }

  /** Revokes the token server-side, then clears it locally regardless. */
  logout(): void {
    const had = this.session !== null;
    this.clear();
    if (had) {
      this.http.post(`${this.baseUrl}/logout`, {}).subscribe({ error: () => {} });
    }
  }

  /** Called by the interceptor when the API rejects our token. */
  clear(): void {
    this.session = null;
    localStorage.removeItem(STORAGE_KEY);
    this.user$.next(null);
  }

  private store(res: AuthResponse): void {
    this.session = {
      token: res.token,
      username: res.username,
      expires_at: res.expires_at,
      isAdmin: res.is_admin,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(this.session));
    this.user$.next(res.username);
  }

  private restore(): StoredSession | null {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw) as StoredSession;
      if (!parsed?.token || !parsed?.username) return null;
      // Drop a session the server would reject anyway, so the app opens on the
      // login form instead of flashing the workspace then 401-ing.
      if (parsed.expires_at && new Date(parsed.expires_at) <= new Date()) {
        localStorage.removeItem(STORAGE_KEY);
        return null;
      }
      return parsed;
    } catch {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
  }
}
