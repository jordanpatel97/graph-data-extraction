function convertToAbsolute(path){
var svgpath = require('svgpath');

var transformed = svgpath(path)
                    .abs()
                    .toString();;
return transformed;}

console.log(convertToAbsolute(process.argv[2]))