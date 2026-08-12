import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpErrorResponse, HttpEventType } from '@angular/common/http';
import {
  ApiService,
  ConversationSummary,
  DocumentInfo,
  IngestResponse,
  UploadResponse,
} from './services/api.service';
import { AuthService } from './services/auth.service';
import { DocumentsComponent } from './components/documents/documents.component';
import {
  AskEvent,
  ChatComponent,
  ChatTurn,
  OpenSourceEvent,
} from './components/chat/chat.component';
import { HistoryComponent } from './components/history/history.component';
import { LoginComponent } from './components/login/login.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, DocumentsComponent, ChatComponent, HistoryComponent, LoginComponent],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent implements OnInit {
  // Documents panel
  documents: DocumentInfo[] = [];
  scanning = false;
  lastIngest: IngestResponse | null = null;
  uploading = false;
  uploadProgress = 0;
  lastUpload: UploadResponse | null = null;
  uploadError: string | null = null;

  // Chat
  turns: ChatTurn[] = [];
  asking = false;
  loadingThread = false;
  activeConversationId: number | null = null;
  activeTitle: string | null = null;

  // History panel
  conversations: ConversationSummary[] = [];
  loadingHistory = false;

  signedIn = false;
  username: string | null = null;

  sourceOpenError: string | null = null;

  constructor(private api: ApiService, private auth: AuthService) {}

  ngOnInit(): void {
    // The interceptor calls auth.clear() on any 401, which lands here and
    // returns the app to the login form without a reload.
    this.auth.user$.subscribe((username) => {
      const wasSignedIn = this.signedIn;
      this.signedIn = username !== null;
      this.username = username;
      if (wasSignedIn && !this.signedIn) {
        this.resetWorkspace();
      }
    });

    if (this.signedIn) {
      this.loadWorkspace();
    }
  }

  onSignedIn(): void {
    this.loadWorkspace();
  }

  signOut(): void {
    this.auth.logout();
  }

  private resetWorkspace(): void {
    this.documents = [];
    this.conversations = [];
    this.turns = [];
    this.activeConversationId = null;
    this.activeTitle = null;
    this.lastIngest = null;
    this.lastUpload = null;
    this.uploadError = null;
  }

  private loadWorkspace(): void {
    this.loadDocuments();
    this.loadConversations();
  }

  // --- Documents -----------------------------------------------------------

  loadDocuments(): void {
    this.api.listDocuments().subscribe({
      next: (docs) => (this.documents = docs),
      error: () => {},
    });
  }

  rescan(): void {
    this.scanning = true;
    this.api.ingestFolder().subscribe({
      next: (res) => {
        this.lastIngest = res;
        this.scanning = false;
        this.loadDocuments();
      },
      error: () => {
        this.scanning = false;
      },
    });
  }

  upload(files: File[]): void {
    this.uploading = true;
    this.uploadProgress = 0;
    this.uploadError = null;
    this.lastUpload = null;

    this.api.uploadDocuments(files).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress) {
          this.uploadProgress = event.total ? Math.round((100 * event.loaded) / event.total) : 0;
        } else if (event.type === HttpEventType.Response) {
          this.lastUpload = event.body;
          this.uploading = false;
          this.loadDocuments();
        }
      },
      error: (err: HttpErrorResponse) => {
        this.uploading = false;
        this.uploadError =
          err.error?.detail ?? err.message ?? 'Upload failed. Is the backend running?';
        this.loadDocuments();
      },
    });
  }

  // --- Chat ----------------------------------------------------------------

  ask({ question, docName }: AskEvent): void {
    this.turns = [...this.turns, { role: 'user', text: question }];
    this.asking = true;

    const startedIn = this.activeConversationId;

    this.api.askQuestion(question, startedIn, docName).subscribe({
      next: (res) => {
        // Ignore a late reply if the user has already switched threads.
        if (this.activeConversationId !== startedIn) {
          this.loadConversations();
          return;
        }
        this.activeConversationId = res.conversation_id;
        this.activeTitle = res.title;
        this.turns = [
          ...this.turns,
          {
            role: 'assistant',
            text: res.answer,
            sources: res.sources,
            provider: res.provider,
            fullDocument: res.full_document,
            truncated: res.truncated,
          },
        ];
        this.asking = false;
        this.loadConversations();
      },
      error: (err: HttpErrorResponse) => {
        this.asking = false;
        if (err.status === 401) return; // interceptor is signing us out
        const message =
          err.status === 409
            ? 'No documents indexed yet - upload a document or rescan the folder first.'
            : err.error?.detail ?? 'Something went wrong reaching the analyzer.';
        this.turns = [...this.turns, { role: 'assistant', text: message, error: true }];
        // The question was saved server-side before generation, so keep the
        // sidebar honest about the thread now existing.
        this.loadConversations();
      },
    });
  }

  /** Opens a chat citation's source file, fetched as a blob so the request
   *  carries the auth header the way `<a href>` never could.
   *
   *  Browsers can only render a handful of types inline (PDF, images, plain
   *  text) - everything else, notably .docx, has no built-in viewer and a
   *  blob tab for one just sits blank or silently downloads. So: viewable
   *  types open in a new tab, everything else is downloaded outright.
   *
   *  For PDFs, `page` (from the citation's chunk metadata) is appended as a
   *  `#page=N` fragment, which Chrome/Edge's built-in PDF viewer jumps to
   *  directly - so the reader lands on the answer instead of page one. */
  openSourceDocument({ docName, page }: OpenSourceEvent): void {
    this.sourceOpenError = null;
    const ext = docName.slice(docName.lastIndexOf('.')).toLowerCase();
    const viewableInline = ['.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.txt', '.md'];

    this.api.getDocumentFile(docName).subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        if (viewableInline.includes(ext)) {
          const target = ext === '.pdf' && page ? `${url}#page=${page}` : url;
          window.open(target, '_blank');
        } else {
          const link = document.createElement('a');
          link.href = url;
          link.download = docName;
          link.click();
        }
        // Give the tab/download time to pick up the blob before releasing it.
        setTimeout(() => URL.revokeObjectURL(url), 60_000);
      },
      error: () => {
        this.sourceOpenError = `Couldn't open "${docName}" - it may have been removed or renamed.`;
      },
    });
  }

  newChat(): void {
    this.turns = [];
    this.activeConversationId = null;
    this.activeTitle = null;
    this.loadingThread = false;
  }

  // --- History -------------------------------------------------------------

  loadConversations(): void {
    this.loadingHistory = this.conversations.length === 0;
    this.api.listConversations().subscribe({
      next: (list) => {
        this.conversations = list;
        this.loadingHistory = false;
        if (this.activeConversationId !== null) {
          const active = list.find((c) => c.id === this.activeConversationId);
          if (active) this.activeTitle = active.title;
        }
      },
      error: () => {
        this.loadingHistory = false;
      },
    });
  }

  openConversation(id: number): void {
    if (id === this.activeConversationId || this.asking) return;

    this.loadingThread = true;
    this.activeConversationId = id;
    this.turns = [];

    this.api.getConversation(id).subscribe({
      next: (detail) => {
        if (this.activeConversationId !== id) return; // switched again mid-flight
        this.activeTitle = detail.title;
        this.turns = detail.messages.map((m) => ({
          role: m.role,
          text: m.text,
          sources: m.sources,
          provider: m.provider,
        }));
        this.loadingThread = false;
      },
      error: () => {
        this.loadingThread = false;
        if (this.activeConversationId === id) {
          this.newChat();
          this.loadConversations();
        }
      },
    });
  }

  deleteConversation(id: number): void {
    this.api.deleteConversation(id).subscribe({
      next: () => {
        this.conversations = this.conversations.filter((c) => c.id !== id);
        if (this.activeConversationId === id) {
          this.newChat();
        }
      },
      error: () => this.loadConversations(),
    });
  }
}
