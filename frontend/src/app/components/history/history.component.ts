import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ConversationSummary } from '../../services/api.service';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './history.component.html',
  styleUrls: ['./history.component.css'],
})
export class HistoryComponent {
  @Input() conversations: ConversationSummary[] = [];
  @Input() activeId: number | null = null;
  @Input() loading = false;
  @Input() username: string | null = null;

  @Output() newChat = new EventEmitter<void>();
  @Output() select = new EventEmitter<number>();
  @Output() remove = new EventEmitter<number>();
  @Output() signOut = new EventEmitter<void>();

  confirmingId: number | null = null;

  askRemove(id: number, event: MouseEvent): void {
    event.stopPropagation(); // don't also load the thread we're deleting
    this.confirmingId = this.confirmingId === id ? null : id;
  }

  confirmRemove(id: number, event: MouseEvent): void {
    event.stopPropagation();
    this.confirmingId = null;
    this.remove.emit(id);
  }

  cancelRemove(event: MouseEvent): void {
    event.stopPropagation();
    this.confirmingId = null;
  }

  /** "3:42 PM" for today, otherwise a short date. */
  when(iso: string): string {
    const at = new Date(iso);
    if (isNaN(at.getTime())) return '';
    const now = new Date();
    const sameDay =
      at.getFullYear() === now.getFullYear() &&
      at.getMonth() === now.getMonth() &&
      at.getDate() === now.getDate();
    return sameDay
      ? at.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
      : at.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }
}
