const menu = document.getElementById('menu')
const close = document.getElementById('close__menu')
const menubar = document.getElementById('menu__bar')
const home = document.querySelector('.home')
menu.addEventListener('click',()=>{
  menubar.style.clipPath='inset(0 0 0 0)'
}
)

close.addEventListener('click',()=>{
  menubar.style.clipPath='inset(0 0 0 100%)'
})