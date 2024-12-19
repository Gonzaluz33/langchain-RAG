import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators } from '@angular/forms';
import { AssistantSearchService } from '../assistant-search.service';

@Component({
  selector: 'app-search-assistant',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './search-assistant.component.html',
  styleUrls: ['./search-assistant.component.css']
})
export class SearchAssistantComponent {
  searchForm: FormGroup;
  loading: boolean = false;
  error: string | null = null;
  assistantResult: any = null;

  constructor(
    private fb: FormBuilder,
    private assistantService: AssistantSearchService
  ) {
    this.searchForm = this.fb.group({
      query: ['', Validators.required],
      keywords: ['']
    });
  }

  onSearch() {
    if (this.searchForm.invalid) {
      return;
    }

    const { query, keywords } = this.searchForm.value;
    this.loading = true;
    this.error = null;
    this.assistantResult = null;

    this.assistantService.searchAssistant(query, keywords).subscribe({
      next: (res) => {
        this.assistantResult = res;
        console.log('Respuesta de la API:', res); // Para depuración
        this.loading = false;
      },
      error: () => {
        this.error = 'Ocurrió un error al obtener la recomendación.';
        this.loading = false;
      }
    });
  }
}

