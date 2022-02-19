function Translate(path, trans){
    var svgpath = require('svgpath');
    
    var transformed = svgpath(path)
                        .transform(trans)
                        .abs()
                        .round(3)
                        .toString();
    return transformed;}
    
    console.log(Translate(process.argv[2],process.argv[3]))