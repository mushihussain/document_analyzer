import { Component, EventEmitter, Input, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  ACCEPTED_EXTENSIONS,
  DocumentInfo,
  IngestResponse,
  UploadedDocument,
  UploadResponse,
} from '../../services/api.service';

@Component({
  selector: 'app-documents',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './documents.component.html',
  styleUrls: ['./documents.component.css'],
})
export class DocumentsComponent {
  @Input() documents: DocumentInfo[] = [];
  @Input() scanning = false;
  @Input() lastIngest: IngestResponse | null = null;
  @Input() uploading = false;
  @Input() uploadProgress = 0;
  @Input() lastUpload: UploadResponse | null = null;
  @Input() uploadError: string | null = null;
  @Input() clearing = false;
  @Output() rescan = new EventEmitter<void>();
  @Output() upload = new EventEmitter<File[]>();
  @Output() clearFolder = new EventEmitter<void>();

  readonly accept = ACCEPTED_EXTENSIONS.join(',');
  dragging = false;
  confirmingClear = false;

  get busy(): boolean {
    return this.scanning || this.uploading || this.clearing;
  }

  askClear(): void {
    this.confirmingClear = true;
  }

  confirmClear(): void {
    this.confirmingClear = false;
    this.clearFolder.emit();
  }

  cancelClear(): void {
    this.confirmingClear = false;
  }

  get uploadedChunkCount(): number {
    return (this.lastUpload?.uploaded ?? []).reduce((sum, doc) => sum + doc.chunks_indexed, 0);
  }

  /** Files that stored fine but produced nothing searchable. */
  get notedUploads(): UploadedDocument[] {
    return (this.lastUpload?.uploaded ?? []).filter((doc) => !!doc.note);
  }

  /** What the last rescan actually did, phrased for the incremental case. */
  get ingestSummary(): string {
    const res = this.lastIngest;
    if (!res) return '';

    const parts: string[] = [];
    if (res.added) parts.push(`${res.added} new`);
    if (res.updated) parts.push(`${res.updated} changed`);
    if (res.removed) parts.push(`${res.removed} removed`);

    if (!parts.length) {
      return res.unchanged
        ? `Already up to date — ${res.unchanged} document(s) unchanged.`
        : 'Nothing to file yet.';
    }

    let summary = `Filed ${parts.join(', ')} (${res.chunks_indexed} passage(s)).`;
    if (res.unchanged) summary += ` ${res.unchanged} unchanged, skipped.`;
    return summary;
  }

  onFilesPicked(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.send(input.files);
    input.value = ''; // let the same file be re-picked after a failed upload
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    if (!this.uploading) {
      this.dragging = true;
    }
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.dragging = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging = false;
    if (!this.uploading) {
      this.send(event.dataTransfer?.files ?? null);
    }
  }

  private send(list: FileList | null): void {
    const files = Array.from(list ?? []);
    if (files.length) {
      this.upload.emit(files);
    }
  }
}
