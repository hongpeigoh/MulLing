import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import Loading from './widget.js'

import { Button } from '@progress/kendo-react-buttons';
import { DropDownList } from '@progress/kendo-react-dropdowns';
import withValueField from './withValueField.jsx';

import Draggable from 'react-draggable';

import createPlotlyComponent from 'react-plotlyjs';
import Plotly from 'plotly.js-cartesian-dist';
const PlotlyComponent = createPlotlyComponent(Plotly);

const DropDownListWithValueField = withValueField(DropDownList)

class SandboxHead extends Component {
    render() {
      return(
        <div className="app-header" id="sandbox-header">
          <div className="app-container container">
            <div className="row textcontent center">
              <h1>Sandbox</h1>
              <h4>Try out our additional functionalities! Create Word Vectors for a chosen language, experiment with the Word Movers' Distance widget, or try out our tokenizer!</h4>
            </div>
          </div>
        </div>
      )
    }
}

class Sandbox extends Component {
    constructor(props) {
        super(props);
        this.state = {
          isLoading: true,
          isComponent: <FastVector/>
        }
    }
    componentDidMount() {
        ReactDOM.render(this.state.isComponent, document.getElementById("field"));
    }
    componentDidUpdate(prevState) {
        if (this.state.isComponent !== prevState.isComponent) {
            ReactDOM.render(this.state.isComponent, document.getElementById("field"));
        }
    }
    render() {
      return(
        <div className="app-container container sandbox width-80" id='app-container'>
            <div className="row textcontent">
                <div className="col-12 col-md-3" style={{paddingTop:"8vh"}}>
                    <div className="row">
                        <div className="col-6 col-md-12" style={{padding: "1px 0vw"}}>
                            <Button
                                primary = {true}
                                icon = "connector"
                                style = {{width: "100%"}}
                                onClick = {()=>this.setState({ isComponent: <FastVector/>})}
                            >FastVector</Button>
                        </div>
                        <div className="col-6 col-md-12" style={{padding: "1px 0vw"}}>
                            <Button
                                primary={true}
                                icon="paint"
                                style={{width: "100%"}}
                                onClick = {()=>this.setState({ isComponent: <Evaluate/>})}
                            >Evaluate</Button>
                        </div>
                        <div className="col-6 col-md-12" style={{padding: "1px 0vw"}}>
                            <Button
                                primary={true}
                                icon="font-family"
                                style={{width: "100%"}}
                                onClick = {()=>this.setState({ isComponent: <Tokenize widget={false}/>})}
                            >Tokenize</Button>
                        </div>
                        <div className="col-6 col-md-12" style={{padding: "1px 0vw"}}>
                            <Button
                                primary={true}
                                icon="word"
                                style={{width: "100%"}}
                                onClick = {()=>this.setState({ isComponent: <WMD widget={false}/>})}
                            >WMD</Button>
                        </div>
                    </div>
                </div>
                <div className="col-12 col-md-9" id="field"/>
            </div>
        </div>
        )
      }
}

class FastVector extends Component {
    constructor(props) {
        super(props);
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleValueChange = this.handleValueChange.bind(this);
        this.state = {
            lang: 'en',
        }
        this.langs = [
            {id:"fr",text:"French"},{id:"la",text:"Latin"},{id:"es",text:"Spanish"},{id:"de",text:"German"},{id:"it",text:"Italian"},{id:"en",text:"English"},{id:"ru",text:"Russian"},{id:"zh",text:"Chinese"},{id:"fi",text:"Finnish"},{id:"pt",text:"Portuguese"},{id:"ja",text:"Japanese"},{id:"nl",text:"Dutch"},{id:"bg",text:"Bulgarian"},{id:"sv",text:"Swedish"},{id:"pl",text:"Polish"},{id:"no",text:"Norwegian Bokmål"},{id:"eo",text:"Esperanto"},{id:"th",text:"Thai"},{id:"sl",text:"Slovenian"},{id:"ms",text:"Malay"},{id:"cs",text:"Czech"},{id:"ca",text:"Catalan"},{id:"ar",text:"Arabic"},{id:"hu",text:"Hungarian"},{id:"se",text:"Northern Sami"},{id:"sh",text:"Serbian"},{id:"el",text:"Greek"},{id:"gl",text:"Galician"},{id:"da",text:"Danish"},{id:"fa",text:"Persian"},{id:"ro",text:"Romanian"},{id:"tr",text:"Turkish"},{id:"is",text:"Icelandic"},{id:"eu",text:"Basque"},{id:"ko",text:"Korean"},{id:"vi",text:"Vietnamese"},{id:"ga",text:"Irish"},{id:"grc",text:"Ancient Greek"},{id:"uk",text:"Ukrainian"},{id:"lv",text:"Latvian"},{id:"he",text:"Hebrew"},{id:"mk",text:"Macedonian"},{id:"ka",text:"Georgian"},{id:"hy",text:"Armenian"},{id:"sk",text:"Slovak"},{id:"lt",text:"Lithuanian"},{id:"ast",text:"Asturian"},{id:"mg",text:"Malagasy"},{id:"et",text:"Estonian"},{id:"oc",text:"Occitan"},{id:"fil",text:"Filipino"},{id:"io",text:"Ido"},{id:"hsb",text:"Upper Sorbian"},{id:"hi",text:"Hindi"},{id:"te",text:"Telugu"},{id:"be",text:"Belarusian"},{id:"fro",text:"Old French"},{id:"sq",text:"Albanian"},{id:"mul",text:"(Multilingual, such as emoji)"},{id:"cy",text:"Welsh"},{id:"xcl",text:"Classical Armenian"},{id:"az",text:"Azerbaijani"},{id:"kk",text:"Kazakh"},{id:"gd",text:"Scottish Gaelic"},{id:"af",text:"Afrikaans"},{id:"fo",text:"Faroese"},{id:"ang",text:"Old English"},{id:"ku",text:"Kurdish"},{id:"vo",text:"Volapük"},{id:"ta",text:"Tamil"},{id:"ur",text:"Urdu"},{id:"sw",text:"Swahili"},{id:"sa",text:"Sanskrit"},{id:"nrf",text:"Norman French"},{id:"non",text:"Old Norse"},{id:"gv",text:"Manx"},{id:"nv",text:"Navajo"},{id:"rup",text:"Aromanian"}];
        this.stopword_bools = [
            {text: "Include", id: "true"},
            {text: "Exclude", id: "false"}];
    };
    handleSubmit(e) {
        e.preventDefault();

        const queryaddress = "http://localhost:5050/fasttext?lang=" + this.state.lang

        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {  
            ReactDOM.render(<Loading/>, document.getElementById('results'));
            if (xhttp.readyState === 4 && xhttp.status === 200){
                var Children = [<div className="center"><Button primary={true} disabled={true}>Download Started</Button></div>];
                console.log(this.response)
                ReactDOM.render(Children, document.getElementById('results'));
            } else if (xhttp.status === 500){
                var EChildren = [<div className="center"><Button disabled={true} look="flat">Error, please reload and try again.</Button></div>];
                ReactDOM.render(EChildren, document.getElementById('results'));
            }
        };
        xhttp.onload = function (e) {
			var blob = xhttp.response;
            var a = document.createElement('a');
            a.href = window.URL.createObjectURL(blob);
            a.download = 'new_wordvecs.txt';
            a.dispatchEvent(new MouseEvent('click'));
		}
        xhttp.open("POST", queryaddress , true);
        xhttp.setRequestHeader('Access-Control-Allow-Origin','*');
        xhttp.responseType = 'blob';
        xhttp.send();
    }
    handleValueChange = (event) => {
        if (event.target.props.name === "lang") {
            this.setState({
                lang: event.target.value
            });
        } else if (event.target.props.name === "stopwords") {
            this.setState({
                lang: event.target.value
            });
        }
    };
    render() {
        return(
            <div>
                <h1>Create FastVector File for Language</h1>
                <form id="tok-input" onSubmit={this.handleSubmit}>
                    <div className="row">
                        <div className="col-3">
                            <h4 style={{paddingTop: "0.5vh"}}>Language: &nbsp;</h4>
                        </div>
                        <div className="col-6">
                            <DropDownListWithValueField
                                    name = "lang"
                                    data = {this.langs}
                                    textField = "text"
                                    valueField = "id"
                                    value = {this.state.lang}
                                    onChange = {this.handleValueChange}
                                    />
                        </div> 
                        <div className="col-3 vcenter">
                            <Button>Download</Button>
                        </div>
                    </div>
                    <div className="row" id="results"/>
                </form>
            </div>
        );
    };
}

class Evaluate extends Component {
    render() {
        return(
            <div>Evaluate</div>
        )
    }
}

class Tokenize extends Component {
    constructor(props) {
        super(props);
        this.textInput = React.createRef();
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleValueChange = this.handleValueChange.bind(this);
        this.state = {
            lang: 'en',
            include_stopwords: 'true'
        }
        this.langs = [
            {text: "English", id: "en"},
            {text: "Chinese", id: "zh"},
            {text: "Malay", id: "ms"},
            {text: "Tamil", id: "ta"}];
        this.stopword_bools = [
            {text: "Include", id: "true"},
            {text: "Exclude", id: "false"}];
    };
    handleSubmit(e) {
        e.preventDefault();
        var self = this;
        const queryaddress = "http://localhost:5050/tokenize?doc=" + this.textInput.current.value + "&lang=" + this.state.lang + "&includestopwords=" + this.state.include_stopwords
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {  
            ReactDOM.render(<Loading/>, document.getElementById('results'));
            if (xhttp.readyState === 4 && xhttp.status === 200){
            var Children = [<div className="col-12">{self.props.widget ===true ? <h4>Tokens:</h4> : <h2>Tokens:</h2>}</div>];
                var Results = JSON.parse(String(this.response)).tokens;
                for (const i in Results) {
                    Children.push(<Snippet key={i} token={Results[i]}/>)
                }
                if (self.props.widget === true){
                    document.getElementById("tok-form").style.display = "None";
                    ReactDOM.render(Children, document.getElementById('results'));
                } else {
                    ReactDOM.render(Children, document.getElementById('results'));
                }
            } 
        };
        xhttp.open("GET", queryaddress , true);
        xhttp.setRequestHeader('Access-Control-Allow-Origin','*');
        xhttp.send();
    }
    handleValueChange = (event) => {
        if (event.target.props.name === "lang") {
            this.setState({
                lang: event.target.value
            });
        } else if (event.target.props.name === "stopwords") {
            this.setState({
                lang: event.target.value
            });
        }
    };
    render() {
        return(
            <div className="col-12">
                <h1>Tokenizer</h1>
                <form id="tok-input" onSubmit={this.handleSubmit}>
                    <div className="row" id="tok-form">
                        <div className="col-9">
                            <div className="row input-form">
                                <input
                                    className="input-bar"
                                    type="text"
                                    placeholder="Phrase 1"
                                    ref={this.textInput}
                                />
                            </div>
                        </div>
                        <div className="col-3 vcenter">
                            <Button>Submit</Button>
                        </div>
                        <div className="col-6">
                            <h4 style={{paddingTop: "0.5vh"}}>Language: &nbsp;</h4>
                            <DropDownListWithValueField
                                    name = "lang"
                                    data = {this.langs}
                                    textField = "text"
                                    valueField = "id"
                                    value = {this.state.lang}
                                    onChange = {this.handleValueChange}
                                    />
                        </div> 
                        <div className="col-6">  
                            <h4 style={{paddingTop: "0.5vh"}}>Stopwords: &nbsp;</h4>
                            <DropDownListWithValueField
                                    name = "stopwords"
                                    data = {this.stopword_bools}
                                    textField = "text"
                                    valueField = "id"
                                    value = {this.state.include_stopwords}
                                    onChange = {this.handleValueChange}
                                    /> 
                        </div> 
                    </div>
                    <div className="row" id="results"/>
                </form>
            </div>
        );
    };
}

function Snippet(props) {
    return(
        <div className="col-2 black-border">{props.token}</div>
    );
}

class WMD extends Component {
    constructor(props) {
        super(props);
        this.textInput1 = React.createRef();
        this.textInput2 = React.createRef();
        this.handleSubmit = this.handleSubmit.bind(this);
        this.handleValueChange = this.handleValueChange.bind(this);
        this.state = {
            lang1: 'en',
            lang2: 'en'
        }
        this.langs = [
            {text: "English", id: "en"},
            {text: "Chinese", id: "zh"},
            {text: "Malay", id: "ms"},
            {text: "Tamil", id: "ta"}];
    };
    handleSubmit(e) {
        e.preventDefault();
        var self = this;
        const queryaddress = "http://localhost:5050/wmd?doc1=" + this.textInput1.current.value + "&doc2=" + this.textInput2.current.value + "&lang1=" + this.state.lang1 + "&lang2=" + this.state.lang2;

        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {  
            ReactDOM.render(<Loading/>, document.getElementById('results'));
            if (xhttp.readyState === 4 && xhttp.status === 200){
                var myResults =  JSON.parse(String(this.response));

                var layout = {
                    title: `Word Movers' Distance: ${myResults.wmd.toFixed(4)}`,
                    annotations: [],
                    showlegend: true,
                    xaxis: {
                        ticks: '',
                        side: 'top',
                        showgrid: true,
                        linecolor: 'black'
                    },
                    yaxis: {
                        ticks: '',
                        ticksuffix: ' ',
                        width: 700,
                        height: 700,
                        autosize: false,
                        showgrid: true,
                        linecolor: 'black'
                    }
                };

                const yValues = myResults.tokens.filter( (token, index) =>{
                    return myResults.pdf1[index] !== 0
                });
                const xValues = myResults.tokens.filter( (token, index) =>{
                    return myResults.pdf2[index] !== 0
                });
                const flow = myResults.flow;
                var zValues = [];

                for ( var i = 0; i < flow.length; i++ ) {
                    if ( myResults.pdf1[i] !== 0) {
                        zValues.push(flow[i].filter( (token, index) =>{
                            return myResults.pdf2[index] !== 0
                        }));

                        for ( var j = 0; j < zValues[i].length; j++ ) {
                            var currentValue = zValues[i][j];
                            var textColor = (currentValue === 0.0 ? 'black' : 'white' )
                            var result = {
                                xref: 'x1',
                                yref: 'y1',
                                x: xValues[j],
                                y: yValues[i],
                                text: zValues[i][j],
                                font: {
                                    family: 'Arial',
                                    size: 12,
                                    color: textColor
                                },
                                showarrow: false,
                            };
                            layout.annotations.push(result);
                        }
                    }
                };

                var data = [{
                    x: xValues,
                    y: yValues,
                    z: zValues,
                    type: 'heatmap',
                    colorscale: [[0, '#ffffff'], [1, '#0000aa']],
                    showscale: false
                }];


                var config = {
                    showLink: false,
                    displayModeBar: true,
                    height: 500,
                    width: 700,
                };
                if (self.props.widget === true){
                    document.getElementById("wmd-form").style.display = "None";
                    ReactDOM.render(<h4>Word Movers' Distance: {myResults.wmd.toFixed(4)}</h4>, document.getElementById('results'));
                } else {
                    ReactDOM.render(<PlotlyComponent className="sandbox-graph" data={data} layout={layout} config={config}/>, document.getElementById('results'));
                }
            } 
        };
        xhttp.open("GET", queryaddress , true);
        xhttp.setRequestHeader('Access-Control-Allow-Origin','*');
        xhttp.send();
    };
    handleValueChange = (event) => {
        if (event.target.props.name === "lang1") {
            this.setState({
                lang1: event.target.value
            });
        } else if (event.target.props.name === "lang2") {
            this.setState({
                lang2: event.target.value
            });
        };
    };
    render() {
        return(
            <div className="col-12">
                <h1>Word Movers' Distance Similarity</h1>
                <form onSubmit={this.handleSubmit}>
                    <div className="row" id="wmd-form">
                        <div className="col-9" id="wmd-input">
                            <div className="input-form">
                                <input
                                    className="input-bar input-text-area"
                                    type="text"
                                    placeholder="Phrase 1"
                                    ref={this.textInput1}
                                />
                                <DropDownListWithValueField
                                    name = "lang1"
                                    data = {this.langs}
                                    textField = "text"
                                    valueField = "id"
                                    value = {this.state.lang1}
                                    onChange = {this.handleValueChange}
                                    />
                            </div>
                            <div className="input-form">
                                <input
                                    className="input-bar input-text-area"
                                    type="text"
                                    placeholder="Phrase 2"
                                    ref={this.textInput2}/>
                                <DropDownListWithValueField
                                    name = "lang2"
                                    data = {this.langs}
                                    textField = "text"
                                    valueField = "id"
                                    value = {this.state.lang2}
                                    onChange = {this.handleValueChange}
                                    />
                            </div>
                        </div>
                        <div className="col-3 vcenter">
                            <Button>Submit</Button>
                        </div>
                    </div>
                    <div className="row" id="results"/>
                </form>
            </div>
        )
    }
}

class PopupWidget extends React.Component {
    constructor( props ) {
        super( props );
        this.state = {
          isComponent: <Tokenize widget={true}/>
        };
    }
    componentDidMount() {
        ReactDOM.render(this.state.isComponent, document.getElementById("popup-field"));
    }
    componentDidUpdate(prevState) {
        if (this.state.isComponent !== prevState.isComponent) {
            ReactDOM.render(this.state.isComponent, document.getElementById("popup-field"));
        }
    }
    render() {
        return (
            <div className="holder">
                <Draggable>
                    <div className="popup">
                        <div className="row">
                            <div className="col-5 center" style={{padding: "0", backgroundColor: "rgb(0,45,135"}}>
                                <Button
                                    primary={true}
                                    icon="font-family"
                                    style={{width: "100%"}}
                                    onClick = {()=>this.setState({ isComponent: <Tokenize widget={true}/>})}
                                >Tokenize</Button>
                            </div>
                            <div className="col-5 center" style={{padding: "0", backgroundColor: "rgb(0,45,135"}}>
                                <Button
                                    primary={true}
                                    icon="word"
                                    style={{width: "100%"}}
                                    onClick = {()=>this.setState({ isComponent: <WMD widget={true}/>})}
                                >WMD</Button>
                            </div>
                            <div className="col-2 center" style={{padding: "0", backgroundColor: "rgb(195,60,60",  boxShadow: "0 3px 1px -2px rgba(0,0,0,0.2), 0 2px 2px 0 rgba(0,0,0,0.14), 0 1px 5px 0 rgba(0,0,0,0.12)"}}>
                                <Button
                                    look="flat"
                                    icon="close"
                                    style={{width: "100%"}}
                                    onClick = {this.props.handler}
                                />
                            </div>
                        </div>
                        <div className="row" id="popup-field"/>
                    </div>
                </Draggable>
            </div>
        );
    }
}




export { Sandbox, SandboxHead, PopupWidget };