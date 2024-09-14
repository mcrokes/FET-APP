window.addEventListener('DOMContentLoaded', async () => {
	await initial_check('regressor');
});

(async()=>{
	const regressorTranslations = await t(["add-regressor"]);

	document.getElementById('regressor-title').innerHTML = regressorTranslations['title'] ?? 'regressor.title';

	const progressTranslations = regressorTranslations['progress'] ?? {};
	if (window.document.getElementById('progress-message')) window.document.getElementById('progress-message').innerHTML = progressTranslations['message-1'] ?? 'regressor.progress.message-1';

	const requiredInfo = window.document.getElementsByClassName('required-field-info');
	if(requiredInfo){
		for (const info of requiredInfo) {
			info.innerHTML = regressorTranslations['required'] ?? 'regressor.required';
		}
		const optionalInfo = window.document.getElementsByClassName('optional-field-info');
		for (const info of optionalInfo) {
			info.innerHTML = regressorTranslations['optional'] ?? 'regressor.optional';
		}
	}

	if (document.getElementById('unit-selector-label')) document.getElementById('unit-selector-label').innerHTML = regressorTranslations['unit'] ?? 'regressor.unit';

	if (document.getElementById('target-selector-label')) document.getElementById('target-selector-label').innerHTML = regressorTranslations['target-selector'] ?? 'regressor.target-selector';
	if (document.getElementById('index-selector-label')) document.getElementById('index-selector-label').innerHTML = regressorTranslations['index-selector'] ?? 'regressor.index-selector';

	const descriptionsTranslations = regressorTranslations['descriptions'] ?? {};
	if (document.getElementById('description-title')) document.getElementById('description-title').innerHTML = descriptionsTranslations['title'] ?? 'regressor.descriptions.title';
	const qVarsTranslations = regressorTranslations['q-vars'] ?? {};
	if (document.getElementById('q-vars-title')) document.getElementById('q-vars-title').innerHTML = qVarsTranslations['title'] ?? 'regressor.q-vars.title';

	const descriptionTexts = window.document.getElementsByClassName('description-text');
	if(descriptionTexts){
		for (const text of descriptionTexts) {
			text.placeholder = descriptionsTranslations['placeholder'] ?? 'regressor.descriptions.placeholder';
		}
		const qVarsText = window.document.getElementsByClassName('q-vars-text');
		for (const text of qVarsText) {
			text.placeholder = `${qVarsTranslations['placeholder'] ?? 'regressor.q-vars.placeholder'}${text.getAttribute('oldValue')}`;
		}
	}


	if (document.getElementById('regressor-cancel-btn')) document.getElementById('regressor-cancel-btn').innerHTML = regressorTranslations['cancel-btn'] ?? 'regressor.cancel-btn';
	if (document.getElementById('regressor-create-back-btn')) document.getElementById('regressor-create-back-btn').innerHTML = regressorTranslations['back-btn'] ?? 'regressor.back-btn';
	if (document.getElementById('regressor-back-btn')) document.getElementById('regressor-back-btn').innerHTML = regressorTranslations['back-btn'] ?? 'regressor.back-btn';
	if (document.getElementById('regressor-accept-btn')) document.getElementById('regressor-accept-btn').innerHTML = regressorTranslations['accept-btn'] ?? 'regressor.accept-btn';
	if (document.getElementById('regressor-next-btn')) document.getElementById('regressor-next-btn').innerHTML = regressorTranslations['next-btn'] ?? 'regressor.next-btn';
	if (document.getElementById('regressor-modify-btn')) document.getElementById('regressor-modify-btn').innerHTML = regressorTranslations['modify-btn'] ?? 'regressor.modify-btn';
	if (document.getElementById('regressor-create-btn')) document.getElementById('regressor-create-btn').innerHTML = regressorTranslations['create-btn'] ?? 'regressor.create-btn';

	if(model_name_field) model_name_field.placeholder = regressorTranslations['name'] ?? 'regressor.name';
	if(model_field_holder) model_field_holder.innerHTML = regressorTranslations['model'] ?? 'regressor.model';
	if(model_description_field) model_description_field.placeholder = regressorTranslations['description'] ?? 'regressor.description';
	if(model_data_set_field_holder) model_data_set_field_holder.innerHTML = regressorTranslations['dataset'] ?? 'regressor.dataset';

	if(document.getElementById('not-available-name')) document.getElementById('not-available-name').innerHTML = regressorTranslations['name-error'] ?? 'regressor.name-error';
	if(document.getElementById('not-regressor-model')) document.getElementById('not-regressor-model').innerHTML = regressorTranslations['model-error'] ?? 'regressor.model-error';
	if(document.getElementById('not-compatible-dataset')) document.getElementById('not-compatible-dataset').innerHTML = regressorTranslations['dataset-error'] ?? 'regressor.dataset-error';

	const defaultUnits = [
		{ id: '', text: 'NaN' },
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
	const units = regressorTranslations['units'] ?? defaultUnits;
	$("#unit-selector").select2({
		tags: true,
		data: units,
	});
	const unit = document.getElementById('unit-selector')
})();