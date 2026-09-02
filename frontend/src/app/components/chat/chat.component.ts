import {
  AfterViewChecked,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  Output,
  ViewChild,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DocumentInfo, SourceSnippet } from '../../services/api.service';

export interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
  sources?: SourceSnippet[];
  error?: boolean;
  /** Provider that produced this answer, when known. */
  provider?: string | null;
  /** True when this answer read one document in full rather than searching. */
  fullDocument?: boolean;
  truncated?: boolean;
  /** True while this turn's answer is still streaming in. */
  streaming?: boolean;
}

export interface AskEvent {
  question: string;
  /** Set to read one document in full instead of searching across all of them. */
  docName: string | null;
}

export interface OpenSourceEvent {
  docName: string;
  /** 1-indexed source page, when known (PDFs only). */
  page: number | null;
}

const PROVIDER_LABELS: Record<string, string> = {
  groq: 'Groq',
  openrouter: 'OpenRouter',
  anthropic: 'Claude',
};

export function providerLabel(name: string): string {
  return PROVIDER_LABELS[name] ?? name;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css'],
})
export class ChatComponent implements AfterViewChecked {
  @Input() turns: ChatTurn[] = [];
  @Input() asking = false;
  @Input() loadingThread = false;
  @Input() title: string | null = null;
  @Input() documents: DocumentInfo[] = [];

  @Output() ask = new EventEmitter<AskEvent>();
  @Output() newChat = new EventEmitter<void>();
  @Output() openSource = new EventEmitter<OpenSourceEvent>();

  @ViewChild('transcript') private transcript?: ElementRef<HTMLElement>;

  question = '';
  /** '' means "search across all documents"; otherwise one document's name. */
  scope = '';
  private lastRenderedCount = -1;

  label(provider: string | null | undefined): string {
    return provider ? providerLabel(provider) : '';
  }

  /** The generic "thinking" placeholder shows only before the real,
   *  incrementally-filling assistant bubble exists - once streaming has
   *  actually started, that bubble replaces it rather than sitting next to it. */
  get showThinkingPlaceholder(): boolean {
    if (!this.asking) return false;
    const last = this.turns[this.turns.length - 1];
    return !(last?.role === 'assistant' && last.streaming);
  }

  ngAfterViewChecked(): void {
    // Stick to the bottom as turns arrive, but leave the user's scroll position
    // alone while they're reading back through a thread.
    const count = this.turns.length + (this.asking ? 1 : 0);
    if (count !== this.lastRenderedCount && this.transcript) {
      this.lastRenderedCount = count;
      const el = this.transcript.nativeElement;
      el.scrollTop = el.scrollHeight;
    }
  }

  submit(): void {
    const question = this.question.trim();
    if (!question || this.asking) return;
    this.question = '';
    this.ask.emit({ question, docName: this.scope || null });
  }
}
