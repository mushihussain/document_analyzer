import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import { AdminUserSummary, ApiService } from '../../services/api.service';

@Component({
  selector: 'app-admin',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './admin.component.html',
  styleUrls: ['./admin.component.css'],
})
export class AdminComponent implements OnInit {
  /** The signed-in admin's own username - own-row actions are disabled to
   *  match the backend, which refuses to let an admin disable/delete themselves. */
  @Input() username: string | null = null;

  @Output() back = new EventEmitter<void>();

  users: AdminUserSummary[] = [];
  loading = true;
  error: string | null = null;

  /** Row currently showing the reset-password form, if any. */
  resettingId: number | null = null;
  newPassword = '';
  resetError: string | null = null;
  resetting = false;

  /** Row currently showing the delete confirm, if any. */
  confirmingDeleteId: number | null = null;
  busyId: number | null = null;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.loading = true;
    this.error = null;
    this.api.listAdminUsers().subscribe({
      next: (users) => {
        this.users = users;
        this.loading = false;
      },
      error: (err: HttpErrorResponse) => {
        this.loading = false;
        this.error = err.error?.detail ?? 'Could not load users.';
      },
    });
  }

  isSelf(user: AdminUserSummary): boolean {
    return !!this.username && user.username.toLowerCase() === this.username.toLowerCase();
  }

  toggleDisabled(user: AdminUserSummary): void {
    this.busyId = user.id;
    this.api.setUserDisabled(user.id, !user.is_disabled).subscribe({
      next: (updated) => {
        this.busyId = null;
        this.users = this.users.map((u) => (u.id === updated.id ? updated : u));
      },
      error: (err: HttpErrorResponse) => {
        this.busyId = null;
        this.error = err.error?.detail ?? 'Could not update that account.';
      },
    });
  }

  startReset(user: AdminUserSummary): void {
    this.resettingId = user.id;
    this.newPassword = '';
    this.resetError = null;
  }

  cancelReset(): void {
    this.resettingId = null;
    this.newPassword = '';
    this.resetError = null;
  }

  confirmReset(user: AdminUserSummary): void {
    if (this.newPassword.length < 8) {
      this.resetError = 'Password must be at least 8 characters.';
      return;
    }
    this.resetting = true;
    this.api.resetUserPassword(user.id, this.newPassword).subscribe({
      next: () => {
        this.resetting = false;
        this.resettingId = null;
        this.newPassword = '';
      },
      error: (err: HttpErrorResponse) => {
        this.resetting = false;
        this.resetError = err.error?.detail ?? 'Could not reset that password.';
      },
    });
  }

  askDelete(user: AdminUserSummary): void {
    this.confirmingDeleteId = user.id;
  }

  cancelDelete(): void {
    this.confirmingDeleteId = null;
  }

  confirmDelete(user: AdminUserSummary): void {
    this.busyId = user.id;
    this.confirmingDeleteId = null;
    this.api.deleteUser(user.id).subscribe({
      next: () => {
        this.busyId = null;
        this.users = this.users.filter((u) => u.id !== user.id);
      },
      error: (err: HttpErrorResponse) => {
        this.busyId = null;
        this.error = err.error?.detail ?? 'Could not delete that account.';
      },
    });
  }
}
