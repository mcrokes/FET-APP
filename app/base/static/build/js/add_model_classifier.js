window.addEventListener('DOMContentLoaded', async () => {
	await initial_check('classifier');
});

(async()=>{
	const classifierTranslations = await t(["add-classifier"]);

	document.getElementById('classifier-title').innerHTML = classifierTranslations['title'] ?? 'classifier.title';

	const progressTranslations = classifierTranslations['progress'] ?? {};
	if (window.document.getElementById('progress-message')) window.document.getElementById('progress-message').innerHTML = progressTranslations['message-1'] ?? 'classifier.progress.message-1';

	const requiredInfo = window.document.getElementsByClassName('required-field-info');
	if(requiredInfo){
		for (const info of requiredInfo) {
			info.innerHTML = classifierTranslations['required'] ?? 'classifier.required';
		}
		const optionalInfo = window.document.getElementsByClassName('optional-field-info');
		for (const info of optionalInfo) {
			info.innerHTML = classifierTranslations['optional'] ?? 'classifier.optional';
		}
	}

	if (document.getElementById('target-selector-label')) document.getElementById('target-selector-label').innerHTML = classifierTranslations['target-selector'] ?? 'classifier.target-selector';
	if (document.getElementById('index-selector-label')) document.getElementById('index-selector-label').innerHTML = classifierTranslations['index-selector'] ?? 'classifier.index-selector';

	const descriptionsTranslations = classifierTranslations['descriptions'] ?? {};
	if (document.getElementById('description-title')) document.getElementById('description-title').innerHTML = descriptionsTranslations['title'] ?? 'classifier.descriptions.title';
	const qVarsTranslations = classifierTranslations['q-vars'] ?? {};
	if (document.getElementById('q-vars-title')) document.getElementById('q-vars-title').innerHTML = qVarsTranslations['title'] ?? 'classifier.q-vars.title';

	const descriptionTexts = window.document.getElementsByClassName('description-text');
	if(descriptionTexts){
		for (const text of descriptionTexts) {
			text.placeholder = descriptionsTranslations['placeholder'] ?? 'classifier.descriptions.placeholder';
		}
		const qVarsText = window.document.getElementsByClassName('q-vars-text');
		for (const text of qVarsText) {
			text.placeholder = `${qVarsTranslations['placeholder'] ?? 'classifier.q-vars.placeholder'}${text.getAttribute('oldValue')}`;
		}
	}


	if (document.getElementById('classifier-cancel-btn')) document.getElementById('classifier-cancel-btn').innerHTML = classifierTranslations['cancel-btn'] ?? 'classifier.cancel-btn';
	if (document.getElementById('classifier-create-back-btn')) document.getElementById('classifier-create-back-btn').innerHTML = classifierTranslations['back-btn'] ?? 'classifier.back-btn';
	if (document.getElementById('classifier-back-btn')) document.getElementById('classifier-back-btn').innerHTML = classifierTranslations['back-btn'] ?? 'classifier.back-btn';
	if (document.getElementById('classifier-accept-btn')) document.getElementById('classifier-accept-btn').innerHTML = classifierTranslations['accept-btn'] ?? 'classifier.accept-btn';
	if (document.getElementById('classifier-next-btn')) document.getElementById('classifier-next-btn').innerHTML = classifierTranslations['next-btn'] ?? 'classifier.next-btn';
	if (document.getElementById('classifier-modify-btn')) document.getElementById('classifier-modify-btn').innerHTML = classifierTranslations['modify-btn'] ?? 'classifier.modify-btn';
	if (document.getElementById('classifier-create-btn')) document.getElementById('classifier-create-btn').innerHTML = classifierTranslations['create-btn'] ?? 'classifier.create-btn';

	if(model_name_field) model_name_field.placeholder = classifierTranslations['name'] ?? 'classifier.name';
	if(model_field_holder) model_field_holder.innerHTML = classifierTranslations['model'] ?? 'classifier.model';
	if(model_description_field) model_description_field.placeholder = classifierTranslations['description'] ?? 'classifier.description';
	if(model_data_set_field_holder) model_data_set_field_holder.innerHTML = classifierTranslations['dataset'] ?? 'classifier.dataset';
})();