import { TestBed } from '@angular/core/testing';

import { AssistantSearchService } from './assistant-search.service';

describe('AssistantSearchService', () => {
  let service: AssistantSearchService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AssistantSearchService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
