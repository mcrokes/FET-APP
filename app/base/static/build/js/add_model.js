const editedModelID = window.location.href.split('/')[5];
let interval = 0;

const add_button = document.getElementById('submit')
const cancel_button = document.getElementById('cancel')

const model_name_field = document.getElementById('model-name')
const model_field = document.getElementById('model')
const model_description_field = document.getElementById('model-description')
const model_data_set_field = document.getElementById('model-data-set')

const model_field_holder = document.getElementById('model-holder')
const model_data_set_field_holder = document.getElementById('model-data-set-holder')

const model_id = document.getElementById('model_id')

let models_list = null;
let editedModelName = null;
const getModelsList = async (model_type) => {
  // get
  response = await fetch(`http://127.0.0.1:5000/INTERNAL_API/${model_type}/namelist`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })

  models_list = await response.json().then((value) => {
    if(editedModelID) {
      editedModelName = value.idPath[editedModelID];
      console.log('editedModelName: ', editedModelName);
    }
    return value.data;
  })
}

const isNameValid = () => editedModelName === model_name_field.value || !models_list.includes(model_name_field.value);

const verify_inputs = (isOnModelCreation = false) => {
  if (isOnModelCreation) {
    add_button.disabled = true;
    add_button.classList.remove("btn-success");
    add_button.classList.add("btn-danger");
    cancel_button.style.display = '';
    cancel_button.disabled = true;
  } else if (
    model_name_field == null ||
    (
      (
        (model_data_set_field == null && model_name_field.value) ||
        (model_name_field.value && model_field.value && model_data_set_field.value)
      ) && isNameValid()
    )
  ) {
    add_button.disabled = false
    add_button.classList.remove('btn-danger');
    add_button.classList.add('btn-success');
    if (model_name_field == null) cancel_button.style.display = '';
    console.log('ACTIVATED');
  } else {
    add_button.disabled = true;
    add_button.classList.remove('btn-success');
    add_button.classList.add('btn-danger');
    cancel_button.style.display = 'none';
    console.log('DEACTIVATED');
  }
  if (model_data_set_field && model_data_set_field.value) {
    model_data_set_field_holder.innerHTML = model_data_set_field.value;
    model_data_set_field_holder.style.color = 'black';
  }
  if (model_field && model_field.value) {
    model_field_holder.innerHTML = model_field.value;
    model_field_holder.style.color = 'black';
  }
}

const initial_check = async (model_type) => {
  if (model_name_field) {
    await getModelsList(model_type);
    console.log('models_list:', models_list);
  }
  if(model_id && model_id.dataset.status == 'Create') {
    verify_inputs(true);
  } else {
    verify_inputs();
  }
}

const is_first_page = () => {
  if (model_id == null || (model_name_field !== null && model_id.dataset.status == 'Create')) {
    verify_inputs();
  }
}

const getProgressPercent = async () => {
  // get
  const progress_bar = document.getElementById('progress_bar');
  const progress_message = document.getElementById('progress-message');

  response = await fetch('http://127.0.0.1:5000/INTERNAL_API/model/percent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: model_id.value
  })

  res = await response.json().then((value) => {
    progress_bar.style.width = `${value[0].percent}%`
    progress_bar.ariaValueNow = value[0].percent
    progress_bar.innerHTML = `${value[0].percent}%`
    progress_message.innerHTML = value[0].message
    console.log(value[0])
    return value[0].percent
  })

  if (res >= 100) {
    clearInterval(interval)
    add_button.disabled = false
    cancel_button.disabled = false
  } else {
    add_button.disabled = true
    cancel_button.disabled = true
  }
}

const manage_percent = async () => {
  if (model_id && model_id.dataset.status == 'Create') {
    interval = setInterval(await getProgressPercent, 3000)
  }
}



window.addEventListener('DOMContentLoaded', manage_percent)
window.addEventListener('input', is_first_page);
