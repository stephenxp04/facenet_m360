const https = require('https')
const fs = require('fs')
const express = require('express')
const path = require('path')
const { get } = require('request')

var options = {
  key: fs.readFileSync('/work/server.key'),
  cert: fs.readFileSync('/work/server.crt'),
  requestCert: false,
  rejectUnauthorized: false
}

const app = express()

app.use(express.json())
app.use(express.urlencoded({ extended: true }))

const viewsDir = path.join(__dirname, '../../face-api.js/examples')
app.use(express.static(path.join(__dirname, '.')))
app.use(express.static(path.join(__dirname, './style')))
app.use(express.static(path.join(viewsDir, '../weights')))
app.use(express.static(path.join(viewsDir, '../dist')))
app.use(express.static(path.join(__dirname, './node_modules/axios/dist')))

app.get('/', (req, res) => res.redirect('/face_detection'))
app.get('/face_detection_video', (req, res) => res.sendFile(path.join(__dirname, 'index.html')))

var server = https.createServer(options,app).listen(8080, () => console.log('Listening on port 8080!'))

function request(url, returnBuffer = true, timeout = 10000) {
  return new Promise(function(resolve, reject) {
    const options = Object.assign(
      {},
      {
        url,
        isBuffer: true,
        timeout,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'
        }
      },
      returnBuffer ? { encoding: null } : {}
    )

    get(options, function(err, res) {
      if (err) return reject(err)
      return resolve(res)
    })
  })
}
