const units = [
  { id: '', text: 'Ninguna' },
  { id: '$', text: '$' },
  { id: '%', text: '%' },
  { id: 'Ltr', text: 'Ltr' },
  { id: 'g', text: 'g' },
  { id: 'kg', text: 'kg' },
  { id: 'cm', text: 'cm' },
  { id: 'dm', text: 'dm' },
  { id: 'm', text: 'm' },
  { id: 'km', text: 'km' },
];
$("#unit-selector").select2({
  tags: true,
  data: units,
});
const unit = document.getElementById('unit-selector')

window.addEventListener('DOMContentLoaded', async () => {
  await initial_check('regressor');
});