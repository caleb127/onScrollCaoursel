window.onload = ()=> {
    const apiKey = "99dcd67cda35586bebdbaab3cee768a0"
    const cityoutput = document.getElementById('cityoutput')
    const add = document.getElementById('add')
    const des = document.getElementById('description')
    var temp = document.getElementById('temp')
    var img = document.getElementById('img')
    var wind = document.getElementById('wind')
    const city = document.getElementById('city')
    var on = document.getElementById('on')
    var off = document.getElementById('off')
    off.addEventListener("click",()=>{
      document.documentElement.classList.remove('dark')
    })
    on.addEventListener("click", ()=>{
      document.documentElement.classList.add('dark')
    })
    var kampala = `https://api.openweathermap.org/data/2.5/weather?q=Kampala&appid=${apiKey}`
    fetch(kampala).then((res)=>res.json()).then(data => {
      cityoutput.innerHTML = "Kampala"
    var desi = data['weather']['0']['description']
    des.innerHTML = desi
    var id = data['weather']['0']['icon']
    var icon = `https://openweathermap.org/img/wn/${id}.png`
                img.src = icon
    var tempature = data['main']['temp']
    temp.innerHTML = eval(`${tempature}-${273}`).toFixed(2) + "°C"
    var windspeed = data['wind']['speed']
    wind.innerHTML = windspeed + "m/s"
    })
    add.addEventListener("click",() =>{
    const getCity = `https://api.openweathermap.org/data/2.5/weather?q=${city.value}&appid=${apiKey}`
    const response = fetch(getCity).then((res) =>  res.json()).then(data => {
      var City = data['name']
      cityoutput.innerHTML  = City
    
    var descrip = data['weather']['0']['description']
    des.innerHTML = descrip
                var tempature = data['main']['temp']
    temp.innerHTML = eval(`${tempature}-${273}`).toFixed(2) + "°C"
                var windspeed = data['wind']['speed']
    wind.innerHTML = windspeed + "m/s"
                var id = data['weather']['0']['icon']
                var icon = `https://openweathermap.org/img/wn/${id}.png`
                img.src = icon
    
                fetch(icon,(res)=>{
                  res.json()
                
                  img.src = icon
                  
                })
    }).catch(()=>{
        cityoutput.innerHTML = "City Not Found"
        wind.innerHTML = ""
        temp.innerHTML = ""
        des.innerHTML = ""
        img.src = ""
        
    })
    })
    if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.theme = 'light'
    localStorage.theme = 'dark'
    localStorage.removeItem('theme')
    }