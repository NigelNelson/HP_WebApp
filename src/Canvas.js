class Canvas extends React.Component {
    render() {
        let img = document.getElementById(this.props.imgID);

        if (img){
            return (
                <div >
                    <canvas id={this.props.id} style={{position:'absolute', left:img.offsetLeft + "px", top:img.offsetTop + "px"}}/>
                </div>
            )
        } else{
            return <canvas id={this.props.id}></canvas>
        }
    }
}