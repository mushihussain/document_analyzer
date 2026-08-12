import { inject } from '@angular/core';
import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { catchError, throwError } from 'rxjs';
import { AuthService } from './auth.service';

/**
 * Attaches the session token to API calls and, if the server rejects it,
 * clears it so the app falls back to the login form instead of looping on 401s.
 */
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(AuthService);
  const token = auth.token;

  // Don't send the old token to the endpoints that mint a new one.
  const isAuthEntry = /\/api\/auth\/(login|register)$/.test(req.url);
  const authed =
    token && !isAuthEntry
      ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } })
      : req;

  return next(authed).pipe(
    catchError((err: HttpErrorResponse) => {
      if (err.status === 401 && !isAuthEntry) {
        auth.clear();
      }
      return throwError(() => err);
    })
  );
};
