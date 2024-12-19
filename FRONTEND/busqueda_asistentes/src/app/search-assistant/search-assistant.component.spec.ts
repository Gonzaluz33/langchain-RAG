import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AssistantSearchService } from '../assistant-search.service';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-search-assistant',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, FormsModule],
  templateUrl: './search-assistant.component.html',
  styleUrls: ['./search-assistant.component.css']
})
export class SearchAssistantComponent {
  searchForm: FormGroup;
  loading: boolean = false;
  error: string | null = null;
  assistantResult: any = null;

  // Variables para manejar las keywords
  keyword: string = '';
  keywords: string[] = [];

  constructor(
    private fb: FormBuilder,
    private assistantService: AssistantSearchService
  ) {
    this.searchForm = this.fb.group({
      query: ['', Validators.required]
    });
  }

  addKeyword() {
    const kw = this.keyword.trim();
    if (kw && !this.keywords.includes(kw)) {
      this.keywords.push(kw);
      this.keyword = '';
    }
  }

  removeKeyword(index: number) {
    this.keywords.splice(index, 1);
  }

  onSearch() {
    if (this.searchForm.invalid) {
      return;
    }

    const { query } = this.searchForm.value;
    this.loading = true;
    this.error = null;
    this.assistantResult = null;

    // Pasamos las keywords como array o coma separadas, según necesites.
    // Suponiendo que la API las acepta como string separado por comas:
    const keywordsStr = this.keywords.join(',');

    this.assistantService.searchAssistant(query, keywordsStr).subscribe({
      next: (res) => {
        this.assistantResult = res;
        console.log('Respuesta de la API:', res);
        this.loading = false;
      },
      error: () => {
        this.error = 'Ocurrió un error al obtener la recomendación.';
        this.loading = false;
      }
    });
  }
}
