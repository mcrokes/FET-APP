const sidebar = document.getElementById('sidebar-container');
const sidebarImage = document.getElementById('sidebar-site-logo');
const topbarImage = document.getElementById('logo-tob-bar');
const sidebarChevronIcon = document.getElementById('chevron-icon-sidebar');
const sidebarItems = document.getElementsByClassName('sidebar-item');
const sidebarIcons = document.getElementsByClassName('sidebar-icon');
const body = document.getElementById('body-general-container');
//const topNav = document.getElementById('top-nav-container');
const sidebarCollapseList = document.getElementById('collapseOne');
const sidebarFooter = document.getElementById('sidebar-footer');
const footer = document.getElementsByTagName('footer')[0];


document.addEventListener('mousemove', (e) => {
	const mouseY = e.clientY;
	const mouseX = e.clientX;
	const windowHeight = window.innerHeight;
	const footerHeight = footer.getBoundingClientRect().height;
	const sidebarWidth = sidebar.getBoundingClientRect().width;
	if (mouseY > windowHeight - 20 && mouseX > sidebarWidth) {
		// Ocultar footer
		footer.style.display = 'block';
	} else if (mouseY < windowHeight - footerHeight) {
		// Mostrar footer
		footer.style.display = 'none';
	}
});

const setSmallSideBar = (mobile = false) => {
	topbarImage.style.display = '';
//	body.style.width = window.innerWidth + 'px';
//	topNav.style.width = window.innerWidth + 'px';
	sidebarImage.style.display = 'none'
	sidebarChevronIcon.style.display = 'none'
	for (const item of sidebarItems) {
		item.style.display = 'none';
	}
	for (const icon of sidebarIcons) {
		icon.style.fontSize = '25px';
		icon.style.marginLeft = '0.5rem';
	}
//	if (!mobile) body.style.width = (window.innerWidth - sidebar.getBoundingClientRect().width + 0.25) + 'px';
//	topNav.style.width = (window.innerWidth - sidebar.getBoundingClientRect().width) + 'px';
	sidebarFooter.style.display = 'block';
	window.localStorage.setItem('sidebar-expanded', 'false');
}

const setNormalSideBar = () => {
	topbarImage.style.display = 'none';
//	body.style.width = window.innerWidth + 'px';
//	topNav.style.width = window.innerWidth + 'px';
	sidebarImage.style.display = 'unset'
	sidebarChevronIcon.style.display = 'unset'
	for (const item of sidebarItems) {
		item.style.display = 'unset';
	}
	for (const icon of sidebarIcons) {
		icon.style.fontSize = '18px';
		icon.style.marginLeft = '0';
	}
//	body.style.width = (window.innerWidth - sidebar.getBoundingClientRect().width + 0.25) + 'px';
//	topNav.style.width = (window.innerWidth - sidebar.getBoundingClientRect().width) + 'px';
	sidebarFooter.style.display = 'flex';
	window.localStorage.removeItem('sidebar-expanded');
}

const hide_sidebar = () => {
	topbarImage.style.display = '';
	sidebar.style.display = 'none';
	window.localStorage.setItem('sidebar-mobile-expanded', 'false');
	window.localStorage.setItem('sidebar-expanded', 'false');
}

const show_sidebar = (onClick = true) => {
	if (window.innerWidth <= 800) {
		if ((sidebar.style.display === 'none' && onClick) || window.localStorage.getItem('sidebar-mobile-expanded') === 'true' && !onClick) {
			sidebar.style.display = 'unset';
			setSmallSideBar(true);
			window.localStorage.setItem('sidebar-mobile-expanded', 'true');
		} else {
			hide_sidebar()
		}
	} else if (sidebarImage.style.display !== 'none' && sidebar.style.display !== 'none') {
		setSmallSideBar();
		window.localStorage.setItem('sidebar-mobile-expanded', 'false');
	} else {
		console.log('OPENING SIDEBAR');
		sidebar.style.display = 'unset';
		setNormalSideBar()
		window.localStorage.setItem('sidebar-mobile-expanded', 'false');
	}
}

if (window.localStorage.getItem('sidebar-expanded') === 'false') {
	show_sidebar(false);
}

const checkPosition = ()=> {
	if (window.innerWidth <= 800 && window.localStorage.getItem('sidebar-mobile-expanded') !== 'true') {
		hide_sidebar();
	} else {
		sidebar.style.display = 'unset';
		if (window.localStorage.getItem('sidebar-mobile-expanded') === 'true' && window.innerWidth <= 800 || window.localStorage.getItem('sidebar-expanded') === 'false') {
			setSmallSideBar();
		} else if(window.innerWidth > 800 && window.localStorage.getItem('sidebar-expanded') !== 'false'){
			setNormalSideBar()
		}
	}
}

window.addEventListener('resize', checkPosition);
window.addEventListener('load', checkPosition);
sidebarCollapseList.addEventListener('transitionend', checkPosition);