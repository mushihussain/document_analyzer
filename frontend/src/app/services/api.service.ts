import { Injectable } from '@angular/core';
import { HttpClient, HttpEvent } from '@angular/common/http';
import { Observable } from 'rxjs';

/** Kept in sync with supported_extensions() in backend/app/ingest.py.
 *  Image types are read with local OCR; they're rejected if OCR_ENABLED=false. */
export const ACCEPTED_EXTENSIONS = [
  '.pdf',
  '.docx',
  '.txt',
  '.md',
  '.png',
  '.jpg',
  '.jpeg',
  '.bmp',
  '.tif',
  '.tiff',
  '.webp',
];

export interface DocumentInfo {
  name: string;
  indexed: boolean;
}

export interface IngestResponse {
  documents_found: number;
  chunks_indexed: number;
  added: number;
  updated: number;
  unchanged: number;
  removed: number;
  failed: SkippedUpload[];
}

export interface UploadedDocument {
  name: string;
  chunks_indexed: number;
  replaced: boolean;
  /** Set when the file stored but yielded nothing searchable. */
  note: string | null;
}

export interface SkippedUpload {
  name: string;
  reason: string;
}

export interface UploadResponse {
  uploaded: UploadedDocument[];
  skipped: SkippedUpload[];
}

export interface ClearResponse {
  removed: number;
}

export interface SourceSnippet {
  doc_name: string;
  excerpt: string;
  /** 1-indexed source page, when known (PDFs only). */
  page: number | null;
}

export interface ChatResponse {
  answer: string;
  sources: SourceSnippet[];
  conversation_id: number;
  title: string;
  /** Which provider in the failover chain answered. */
  provider: string;
  model: string;
  /** True when this answer used whole-document mode instead of search. */
  full_document: boolean;
  /** Set when full_document is true and the document had to be sampled down to fit. */
  truncated: boolean;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  sources: SourceSnippet[];
  provider: string | null;
}

export interface ProviderStatus {
  name: string;
  model: string;
  configured: boolean;
  /** Position in the live chain, or 0 if skipped for a missing key. */
  order: number;
}

export interface ConversationSummary {
  id: number;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationDetail {
  id: number;
  title: string;
  messages: ChatMessage[];
}

export interface AdminUserSummary {
  id: number;
  username: string;
  created_at: string;
  is_admin: boolean;
  is_disabled: boolean;
  document_count: number;
  conversation_count: number;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  // Relative, not absolute - Nginx (prod) or the dev-server proxy (see
  // proxy.conf.json) forwards this to the backend, so the same build works
  // whether it's opened from localhost or the deployed domain.
  private readonly baseUrl = 'api';

  constructor(private http: HttpClient) {}

  listDocuments(): Observable<DocumentInfo[]> {
    return this.http.get<DocumentInfo[]>(`${this.baseUrl}/documents`);
  }

  ingestFolder(): Observable<IngestResponse> {
    return this.http.post<IngestResponse>(`${this.baseUrl}/ingest`, {});
  }

  /** Deletes every file in the scanned folder and wipes the index. Irreversible. */
  clearDocuments(): Observable<ClearResponse> {
    return this.http.post<ClearResponse>(`${this.baseUrl}/documents/clear`, {});
  }

  /** Deletes one document: its file, its indexed chunks, and its scan bookkeeping. */
  deleteDocument(name: string): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/documents/${encodeURIComponent(name)}`);
  }

  /** Emits upload-progress events, then the final response. */
  uploadDocuments(files: File[]): Observable<HttpEvent<UploadResponse>> {
    const form = new FormData();
    for (const file of files) {
      form.append('files', file, file.name);
    }
    return this.http.post<UploadResponse>(`${this.baseUrl}/documents/upload`, form, {
      reportProgress: true,
      observe: 'events',
    });
  }

  /** Omit `conversationId` to start a new thread. Pass `docName` to ask
   *  against one document's full text instead of searching across all of them. */
  askQuestion(
    question: string,
    conversationId: number | null,
    docName: string | null = null
  ): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/chat`, {
      question,
      conversation_id: conversationId,
      doc_name: docName,
    });
  }

  /** Same as askQuestion, but the server streams the answer as newline-
   *  delimited JSON instead of one blocking response. `reportProgress` is
   *  what makes Angular emit a DownloadProgress event (with the response
   *  text downloaded *so far*, in `partialText`) as each chunk of the body
   *  arrives, rather than only once the whole thing has finished - that's
   *  the piece that makes this actually stream instead of just being a
   *  slower version of askQuestion. */
  streamChat(
    question: string,
    conversationId: number | null,
    docName: string | null = null
  ): Observable<HttpEvent<string>> {
    return this.http.post(
      `${this.baseUrl}/chat/stream`,
      { question, conversation_id: conversationId, doc_name: docName },
      { responseType: 'text', reportProgress: true, observe: 'events' }
    );
  }

  /** Raw bytes of one indexed document, for opening a chat citation. */
  getDocumentFile(name: string): Observable<Blob> {
    return this.http.get(`${this.baseUrl}/documents/${encodeURIComponent(name)}/file`, {
      responseType: 'blob',
    });
  }

  // --- Admin (requires an admin account; see ADMIN_USERNAMES) --------------

  listAdminUsers(): Observable<AdminUserSummary[]> {
    return this.http.get<AdminUserSummary[]>(`${this.baseUrl}/admin/users`);
  }

  setUserDisabled(userId: number, disabled: boolean): Observable<AdminUserSummary> {
    return this.http.patch<AdminUserSummary>(`${this.baseUrl}/admin/users/${userId}`, {
      disabled,
    });
  }

  resetUserPassword(userId: number, newPassword: string): Observable<void> {
    return this.http.post<void>(`${this.baseUrl}/admin/users/${userId}/reset-password`, {
      new_password: newPassword,
    });
  }

  /** Deletes the account, their conversations, documents, and vector index. Irreversible. */
  deleteUser(userId: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/admin/users/${userId}`);
  }

  listProviders(): Observable<ProviderStatus[]> {
    return this.http.get<ProviderStatus[]>(`${this.baseUrl}/providers`);
  }

  listConversations(): Observable<ConversationSummary[]> {
    return this.http.get<ConversationSummary[]>(`${this.baseUrl}/conversations`);
  }

  getConversation(id: number): Observable<ConversationDetail> {
    return this.http.get<ConversationDetail>(`${this.baseUrl}/conversations/${id}`);
  }

  renameConversation(id: number, title: string): Observable<ConversationSummary> {
    return this.http.patch<ConversationSummary>(`${this.baseUrl}/conversations/${id}`, { title });
  }

  deleteConversation(id: number): Observable<void> {
    return this.http.delete<void>(`${this.baseUrl}/conversations/${id}`);
  }
}
