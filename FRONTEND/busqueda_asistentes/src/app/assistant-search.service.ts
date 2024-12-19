// src/app/services/assistant-search.service.ts
import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class AssistantSearchService {
  private apiUrl = 'http://127.0.0.1:8000/query';

  constructor(private http: HttpClient) {}

  searchAssistant(query: string, keywords?: string): Observable<any> {
    const body: any = { query };
    if (keywords && keywords.trim()) {
      body.keywords = keywords;
    }

    const headers = new HttpHeaders({
      'Content-Type': 'application/json' // Especificar que el cuerpo es JSON
    });

    return this.http.post<any>(this.apiUrl, body, { headers });
  }
}


