// src/app/app.component.ts
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SearchAssistantComponent } from './search-assistant/search-assistant.component';

@Component({
  standalone: true,
  imports: [CommonModule, SearchAssistantComponent],
  selector: 'app-root',
  template: `<app-search-assistant></app-search-assistant>`
})
export class AppComponent {}
