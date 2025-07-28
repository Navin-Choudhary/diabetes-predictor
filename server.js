const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const path = require('path');

const app = express();
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.render('form');
});

app.post('/predict', async (req, res) => {
    const input = [
        parseFloat(req.body.pregnancies),
        parseFloat(req.body.glucose),
        parseFloat(req.body.bloodPressure),
        parseFloat(req.body.skinThickness),
        parseFloat(req.body.insulin),
        parseFloat(req.body.bmi),
        parseFloat(req.body.dpf),
        parseFloat(req.body.age)
    ];

    try {
        const response = await axios.post('http://localhost:5000/predict', {
            features: input
        });

        const result = response.data.prediction;
        res.render('result', { prediction: result });
    } catch (err) {
        console.error(err);
        res.send('Error predicting');
    }
});

app.listen(3000, () => {
    console.log('Server started on http://localhost:3000');
});
