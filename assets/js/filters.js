let activeFilters = { providers: new Set(), cost: new Set(), categories: new Set() };

function initFilters() {
  document.querySelectorAll('.chip[data-filter]').forEach(chip => {
    chip.addEventListener('click', () => toggleFilter(chip));
  });

  const clearBtn = document.getElementById('clear-filters');
  if (clearBtn) clearBtn.addEventListener('click', clearAllFilters);
}

function toggleFilter(chip) {
  const filterType = chip.dataset.filter;
  const value = chip.dataset.value;

  if (activeFilters[filterType].has(value)) {
    activeFilters[filterType].delete(value);
    chip.classList.remove('chip--active');
  } else {
    activeFilters[filterType].add(value);
    chip.classList.add('chip--active');
  }

  store.set('filters.providers', [...activeFilters.providers]);
  store.set('filters.cost', [...activeFilters.cost]);
  store.set('filters.categories', [...activeFilters.categories]);

  applyFilters();
  updateFilterCount();
}

function clearAllFilters() {
  activeFilters = { providers: new Set(), cost: new Set(), categories: new Set() };
  document.querySelectorAll('.chip--active').forEach(c => c.classList.remove('chip--active'));
  store.set('filters.providers', []);
  store.set('filters.cost', []);
  store.set('filters.categories', []);
  applyFilters();
  updateFilterCount();
}

function applyFilters() {
  const cards = document.querySelectorAll('[data-filterable]');
  let visibleCount = 0;

  cards.forEach(card => {
    const providers = (card.dataset.providers || '').split(',').filter(Boolean);
    const cost = card.dataset.cost || '';
    const categories = (card.dataset.categories || '').split(',').filter(Boolean);

    const providerMatch = activeFilters.providers.size === 0 ||
      [...activeFilters.providers].some(p => providers.includes(p));
    const costMatch = activeFilters.cost.size === 0 ||
      activeFilters.cost.has(cost);
    const categoryMatch = activeFilters.categories.size === 0 ||
      [...activeFilters.categories].some(c => categories.includes(c));

    const visible = providerMatch && costMatch && categoryMatch;

    if (visible) {
      card.classList.remove('card--hidden');
      visibleCount++;
    } else {
      card.classList.add('card--hidden');
    }
  });

  const noResults = document.getElementById('no-results');
  if (noResults) noResults.style.display = visibleCount === 0 ? 'block' : 'none';
}

function updateFilterCount() {
  const total = activeFilters.providers.size + activeFilters.cost.size + activeFilters.categories.size;
  const countEl = document.getElementById('filter-count');
  if (countEl) {
    countEl.textContent = total > 0 ? `${total} active` : '';
    countEl.style.display = total > 0 ? 'inline' : 'none';
  }
  const clearBtn = document.getElementById('clear-filters');
  if (clearBtn) clearBtn.style.display = total > 0 ? 'inline-flex' : 'none';
}

function restoreFilters() {
  const savedProviders = store.get('filters.providers') || [];
  const savedCost = store.get('filters.cost') || [];
  const savedCategories = store.get('filters.categories') || [];

  [...savedProviders, ...savedCost, ...savedCategories].forEach(value => {
    const chip = document.querySelector(`.chip[data-value="${value}"]`);
    if (chip) {
      const filterType = chip.dataset.filter;
      activeFilters[filterType].add(value);
      chip.classList.add('chip--active');
    }
  });

  applyFilters();
  updateFilterCount();
}
