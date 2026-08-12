import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
})
export class LoginComponent {
  @Output() signedIn = new EventEmitter<void>();

  mode: 'login' | 'register' = 'login';
  username = '';
  password = '';
  confirm = '';
  busy = false;
  error: string | null = null;

  readonly minPasswordLength = 8;

  constructor(private auth: AuthService) {}

  get isRegister(): boolean {
    return this.mode === 'register';
  }

  get submitLabel(): string {
    if (this.busy) return this.isRegister ? 'Creating…' : 'Signing in…';
    return this.isRegister ? 'Create account' : 'Sign in';
  }

  switchMode(): void {
    this.mode = this.isRegister ? 'login' : 'register';
    this.error = null;
    this.password = '';
    this.confirm = '';
  }

  submit(): void {
    if (this.busy) return;

    const username = this.username.trim();
    if (!username || !this.password) {
      this.error = 'Enter a username and password.';
      return;
    }
    if (this.isRegister) {
      if (this.password.length < this.minPasswordLength) {
        this.error = `Password must be at least ${this.minPasswordLength} characters.`;
        return;
      }
      if (this.password !== this.confirm) {
        this.error = 'Passwords do not match.';
        return;
      }
    }

    this.busy = true;
    this.error = null;

    const request = this.isRegister
      ? this.auth.register(username, this.password)
      : this.auth.login(username, this.password);

    request.subscribe({
      next: () => {
        this.busy = false;
        this.password = '';
        this.confirm = '';
        this.signedIn.emit();
      },
      error: (err: HttpErrorResponse) => {
        this.busy = false;
        this.error =
          err.error?.detail ??
          (err.status === 0
            ? 'Cannot reach the server. Is the backend running on port 8000?'
            : 'Something went wrong. Try again.');
      },
    });
  }
}
