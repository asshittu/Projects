'use strict';

const header = document.querySelector('.site-header');
const navigation = document.getElementById('site-nav');
const menuButton = header?.querySelector('.nav-toggle');

if (navigation && menuButton) {
  const mobile = window.matchMedia('(max-width: 720px)');

  function closeMenu(returnFocus = false) {
    const wasOpen = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-expanded', 'false');
    if (returnFocus && wasOpen) menuButton.focus();
  }

  header.classList.add('is-enhanced');
  menuButton.hidden = false;
  menuButton.addEventListener('click', () => {
    const isOpen = menuButton.getAttribute('aria-expanded') === 'true';
    menuButton.setAttribute('aria-expanded', String(!isOpen));
  });
  mobile.addEventListener('change', () => closeMenu());
  header.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && mobile.matches) closeMenu(true);
  });
  navigation.addEventListener('click', (event) => {
    const link = event.target.closest('a[href]');
    if (!link || !mobile.matches) return;
    closeMenu();
    const target = link.hash ? document.getElementById(link.hash.slice(1)) : null;
    if (target && link.pathname === location.pathname) {
      target.tabIndex = -1;
      target.focus({ preventScroll: true });
    }
  });
}

const filters = document.getElementById('project-filters');
const projectCount = document.getElementById('project-count');

if (filters && projectCount) {
  const buttons = Array.from(filters.querySelectorAll('button[data-filter]'));
  const projects = Array.from(document.querySelectorAll('#project-library .project-card'));
  const categories = projects.map((project) => project.dataset.categories.split(' '));

  filters.hidden = false;
  filters.addEventListener('click', (event) => {
    const selected = event.target.closest('button[data-filter]');
    if (!selected) return;
    const filter = selected.dataset.filter;
    buttons.forEach((button) => button.setAttribute('aria-pressed', String(button === selected)));
    let shown = 0;
    projects.forEach((project, index) => {
      const visible = filter === 'all' || categories[index].includes(filter);
      project.hidden = !visible;
      if (visible) shown += 1;
    });
    projectCount.textContent = `${shown} ${shown === 1 ? 'project' : 'projects'}`;
  });
}
