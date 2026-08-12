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

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly baseUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) {}

  listDocuments(): Observable<DocumentInfo[]> {
    return this.http.get<DocumentInfo[]>(`${this.baseUrl}/documents`);
  }

  ingestFolder(): Observable<IngestResponse> {
    return this.http.post<IngestResponse>(`${this.baseUrl}/ingest`, {});
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

  /** Raw bytes of one indexed document, for opening a chat citation. */
  getDocumentFile(name: string): Observable<Blob> {
    return this.http.get(`${this.baseUrl}/documents/${encodeURIComponent(name)}/file`, {
      responseType: 'blob',
    });
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
